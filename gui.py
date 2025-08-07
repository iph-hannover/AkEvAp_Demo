import os
import time
import cv2, imutils
import imageio
import sys
import queue
import threading
import numpy as np
import pandas as pd
from enum import Enum
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify

import utils
from collections import Counter
from skeletons import get_bone_colors, get_joint_colors, BONES, JOINTS, COLORS
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QListWidgetItem
from PyQt5.QtGui import QImage, QPixmap, QColor
from bibs.guis import gui_main
from bibs.functions import userInputs
from niosh import NIOSH, CouplingType
from pose_classification import LMMClassificator, PoseTypeLMM

NO_GUI = False  # useful for batch evaluation of pipeline on multiple videos
RECORD_GUI = True
RESULT_DIR = "./tmp/"
SRT_STREAM = "srt://127.0.0.1:8080?streamid=myserver/stream/test"
# SRT_STREAM = 0  # use webcam
IMG_FOLDER = "./frames/color_%8d.jpg"  # read from frames (e.g. extracted using ffmpeg)
IMG_FOLDER = "./video.mp4"             # read from video file
START_IDX = 0  # change this to start at a specific frame for debugging
INITIAL_MODE = 'CAPTURE'  # 'CAPTURE' or 'REPLAY' in order to start in Mode.REPLAY
LOOP_VIDEO = True
FORCE_TRAM = False          # force tram evaluation at every frame
USE_HAND_CONTACT = True     # always required for NIOSH evaluation
VERBOSE = False

if USE_HAND_CONTACT:
    from hand_contact import ContactDetector
    from contact_postprocessing import ContactPostprocessor
from pose_tram import PoseEstimator
from pose_gt import PoseGTLoader

if RECORD_GUI:
    import mss

class Mode(Enum):
    CAPTURE = 1
    REPLAY = 2

class AkevapVisual(QWidget, gui_main.Ui_Form):
    def __init__(self):
        super(AkevapVisual, self).__init__()

        if not NO_GUI:
            self.setupUi(self)  # call gui_main.Ui_Form.setupUi()

            self.listWi_Lifts.clear()
            self.label.setStyleSheet("")
            self.btn_start.clicked.connect((lambda: self.show_clip(self.hSlid_VideoMoverVon.value(), self.hSlid_VideoMoverBis.value())))
            self.hSlid_VideoMoverBis.valueChanged.connect(self.end_slider_changed)
            self.hSlid_VideoMoverVon.valueChanged.connect(self.start_slider_changed)
            self.dial_frameSpeed.valueChanged.connect(self.speed_changed)
            self.btn_pause.clicked.connect(self.pause_video)
            self.btn_stop.clicked.connect(self.stop_video)
            self.btn_nexFrame.clicked.connect(self.show_next_frame)
            self.btn_preFrame.clicked.connect(self.show_prev_frame)
            self.listWi_Lifts.itemClicked.connect(self.select_lift)
            self.btn_stopRecod.clicked.connect(self.switch_mode)
            self.btn_showLMM.clicked.connect(self.openLMM)
            self.lineE_liftGewicht.textChanged.connect(self.changeLiftGewicht)

        self.mode = Mode.CAPTURE
        if INITIAL_MODE == 'REPLAY':
            self.switch_mode()
        self.skip_frames = 1  # default: show every frame in Mode.CAPTURE
        self.init_capture()
        self.first_frame_time = None
        self.frame_count = START_IDX
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)

        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if NO_GUI:
            self.img_width = frame_width
            self.img_height = frame_height
        else:
            self.size_multiplier = min((self.label.width() - self.label.width() % 2) / frame_width,
                                       (self.label.height() - self.label.height() % 2) / frame_height)  # ensure width and height are even, otherwise libx264 will crash
            self.img_width = int(frame_width * self.size_multiplier)
            self.img_height = int(frame_height * self.size_multiplier)

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.writer_video = imageio.get_writer(Path(RESULT_DIR, 'video_2d.mp4'), fps=fps, **dict(codec='h264', ffmpeg_params=['-crf', '17'], macro_block_size=None))
        if RECORD_GUI:
            self.writer_gui = None

        self.skeleton_type = '2D'
        self.server = Flask(__name__)
        self.server_thread_run = False
        self.server_thread = None
        self.setup_flask_routes()
        self.points_queue = queue.Queue()

        self.thread_run = False
        self.thread = None

        self.CACHE = {}
        self.CACHE_df = None  # only available if video recording is stopped
        self.update = False
        self.play = False
        self.playback_timer = None

        if USE_HAND_CONTACT:
            print(f'Loading hand contact estimator')
            self.contact_detector = ContactDetector(self.skeleton_type, initial_frame_idx=self.frame_count)
            self.contact_postprocessor = ContactPostprocessor(self.contact_detector, self.CACHE)
            self.num_lifts = 0
        print(f'Loading human pose estimator')
        self.pose_estimator = PoseEstimator()
        self.person_detected_count = 0
        self.pose_loader = PoseGTLoader(Path(IMG_FOLDER).parent, initial_frame_idx=self.frame_count, detector3D='apple')
        self.pose_loader_hq = PoseGTLoader(Path(IMG_FOLDER).parent, initial_frame_idx=self.frame_count, detector3D='tram')
        self.person_detected = False

        self.NIOSH = NIOSH(self.skeleton_type, load_weight=15.0, initial_frame_count=self.frame_count, coupling_type=CouplingType.FAIR)
        print(f'Loading pose classificator')
        self.pose_classificator = LMMClassificator()

    def init_capture(self):
        print("Startup time for VideoCapture may take up to 30s or fail => retry")
        self.cap = cv2.VideoCapture(SRT_STREAM)
        self.cap_src = 'stream'
        if not self.cap.isOpened():
            print(f"Stream {SRT_STREAM} not available. Reading local frames from {IMG_FOLDER}")
            self.cap = cv2.VideoCapture(IMG_FOLDER)
            self.cap_src = 'folder'
            print(f"Got {int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames")
            if not NO_GUI:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.init_dial(fps)
        else:
            print(f"Using stream {SRT_STREAM}")
        self.is_webcam = (self.cap_src == 'stream' and SRT_STREAM == "0")

    def init_dial(self, fps):
        self.lab_diaLinks.setText('Normal')
        self.lab_diaMitte.setText('Schnell')
        self.lab_diaRechts.setText('Extrem')
        self.dial_frameSpeed.setValue(self.skip_frames)
        self.dial_frameSpeed.setRange(1, int(fps))
        self.dial_frameSpeed.setEnabled(True)

    def changeLiftGewicht(self):
        if self.lineE_liftGewicht.text() == '':
            newGewicht = 1
        else:
            newGewicht = int(self.lineE_liftGewicht.text())
        self.NIOSH.load_weight = newGewicht

    def switch_mode(self):
        if self.mode == Mode.CAPTURE:
            # closing the video writer here will not work properly as we are use it from different threads (events handled by app.exec_() event loop and the main loop of the worker thread (update_thread))
            # => just set a flag here and close it in the main loop (update_thread)
            # alternative: use signaling to notify writer about new frames and let all image processing and video writing run in the event loop started by app.exec_()
            self.mode = Mode.REPLAY
            self.btn_stopRecod.setText("Start Recording")
            self.dial_frameSpeed.setEnabled(True)
            self.btn_nexFrame.setEnabled(True)
            self.btn_pause.setEnabled(True)
            self.btn_preFrame.setEnabled(True)
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(True)
            self.hSlid_VideoMoverBis.setEnabled(True)
            self.hSlid_VideoMoverVon.setEnabled(True)
            self.btn_stopRecod.setEnabled(False)
            self.btn_showLMM.setEnabled(True)

        elif self.mode == Mode.REPLAY:
            self.mode = Mode.CAPTURE
            self.btn_stopRecod.setText("Stop Recording")
            self.init_capture()
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.writer_video = imageio.get_writer(Path(RESULT_DIR, 'video_2d.mp4'), fps=fps, **dict(codec='h264', ffmpeg_params=['-crf', '17'], macro_block_size=None))
            self.CACHE = {}
            self.CACHE_df = None  # only available if video recording is stopped

    def openLMM(self):
        liftTyps = []
        liftNrs = self.listWi_Lifts.count()//2 + 1
        for nr in range(liftNrs):
            pose_typeStart = self.CACHE_df[(self.CACHE_df['num_lifts'] == nr+1) & (self.CACHE_df['contact_state'] == 'carry')].iloc[0]["pose_type"]
            pose_typeEnd = self.CACHE_df[(self.CACHE_df['num_lifts'] == nr+1) & (self.CACHE_df['contact_state'] == 'carry')].iloc[-1]["pose_type"]
            liftTyps.append([pose_typeStart, pose_typeEnd])
        # liftTyps.append([0, 1])
        haeufigkeiten = Counter(tuple(pair) for pair in liftTyps)
        if len(haeufigkeiten) == 1:
            mainElement = next(iter(haeufigkeiten.items()))
            mainLift = list(mainElement[0])
        else:
            iterator = iter(haeufigkeiten.items())
            firstElement = next(iterator)
            secElement = next(iterator)
            if secElement[1] / firstElement[1] > 0.05:
                mainLift = [4, 4]
            else:
                mainLift = list(firstElement[0])
        self.user = userInputs.userwidget(liftNrs, mainLift)
        # self.user.finish.connect(self.backEmit)
        self.user.show()
        test = 7

    def load_dataframe(self, df):
        self.play = False
        self.CACHE_df = df

        # init sliders
        min_ = self.CACHE_df['frame'].iloc[0]
        max_ = self.CACHE_df['frame'].iloc[-1]
        self.hSlid_VideoMoverBis.setRange(min_, max_)
        self.hSlid_VideoMoverBis.setValue(max_)
        self.lab_frameBis.setText(f'bis {max_}')
        self.hSlid_VideoMoverVon.setRange(min_, max_)
        self.hSlid_VideoMoverVon.setValue(min_)
        self.lab_framVon.setText(f'von {min_}')
        self.lineE_FrameSpeed.setText(str(self.dial_frameSpeed.value()))

        # init lift list
        self.listWi_Lifts.clear()
        for id in self.CACHE_df['num_lifts'].unique():
            if id == 0:
                continue
            chunk_start, chunk_end = self.get_lifting_chunk(id)

            # start frame
            frame_data = self.CACHE_df.loc[chunk_start]
            assert id == frame_data['num_lifts']
            assert 'lift' == frame_data['contact_state_change']
            liftColor = self.getColor(frame_data['NIOSH_LI'])
            pose_type = PoseTypeLMM(frame_data['pose_type'])
            item = QListWidgetItem(f"Lift {id} ({'lift'})\tframe: {chunk_start}\ttype: {pose_type.name} [{pose_type.value}]")
            item.setBackground(QColor(liftColor))
            self.listWi_Lifts.addItem(item)

            # end frame
            frame_data = self.CACHE_df.loc[chunk_end+1]
            assert id == frame_data['num_lifts']
            assert 'drop' == frame_data['contact_state_change']
            liftColor = self.getColor(frame_data['NIOSH_LI'])
            pose_type = PoseTypeLMM(frame_data['pose_type'])
            item = QListWidgetItem(f"Lift {id} ({'drop'})\tframe: {chunk_end+1}\ttype: {pose_type.name} [{pose_type.value}]")
            item.setBackground(QColor(liftColor))
            self.listWi_Lifts.addItem(item)

    def update_gui(self):
        contact_change_idx = self.CACHE[self.frame_count]['computed_contact_state_change_at']
        contact_state_change = self.CACHE[self.frame_count]['computed_contact_state_change']
        frame_data = self.CACHE[contact_change_idx]
        assert contact_state_change == frame_data['contact_state_change']
        num_lifts = self.listWi_Lifts.count()//2 + 1
        assert num_lifts == frame_data['num_lifts']
        liftColor = self.getColor(frame_data['NIOSH_LI'])
        pose_type = PoseTypeLMM(frame_data['pose_type'])
        item = QListWidgetItem(f"Lift {num_lifts} ({contact_state_change})\tframe: {contact_change_idx}\ttype: {pose_type.name} [{pose_type.value}]")
        item.setBackground(QColor(liftColor))
        self.listWi_Lifts.addItem(item)
        self.set_lift_data(frame_data)

    def set_lift_data(self, frame_data):
        fields = [('lineE_horiPos', 'HM', 'horizontal location (H)'),
                  ('lineE_vertiPos', 'VM', 'vertical location (V)'),
                  ('lineE_vertiTravel', 'DM', 'vertical travel distance (D)'),
                  ('lineE_asyDeg', 'AM', 'asymmetry angle (A)'),
                  ('lineE_liftSeq', 'FM', 'lifting frequency (F)'),  # TODO: work duration (WD), vertical location (V_start)
                  ('lineE_dogging', 'CM', 'coupling type (CT)')]  # TODO: vertical location (V)

        for field_name, name, info_key in fields:
            data_field = getattr(self, f"{field_name}_data")
            score_field = getattr(self, f"{field_name}_score")
            try:
                data_info = frame_data[f'NIOSH_{name}']['info'][info_key]
                score = frame_data[f'NIOSH_{name}']['value']
            except:
                data_info = "n/A"
                score = None
            data_field.setText(data_info if isinstance(data_info, str) else f"{data_info[0]:.1f}")  # contains either "n/A" or tuple(value, unit)
            score_field.setText("n/A" if score is None else f"{score:.2f}")  # contains either None or a scalar value

        liftIndex = round(frame_data['NIOSH_LI'], 2) if frame_data['NIOSH_LI'] != np.inf else np.inf
        liftColor = self.getColor(liftIndex)
        self.lineE_LI.setText(f"{liftIndex:.2f}")
        self.btnLicht.setStyleSheet("background-color:" + liftColor)
        self.lineE_RWL.setText(f"{frame_data['NIOSH_RWL']:.2f}")

    def getColor(self, lifting_index):
        if lifting_index > 3:
            liftColor = 'red'
        elif 2.1 <= lifting_index < 3:
            liftColor = 'orange'
        elif 1.5 <= lifting_index < 2.1:
            liftColor = 'yellow'
        elif 1.1 <= lifting_index < 1.5:
            liftColor = 'limegreen'
        else:
            liftColor = 'green'
        return liftColor

    def get_lifting_chunk(self, lift_id):
        """
        Get the start and end frame indices for a given lift

        Parameters
        ----------
        lift_id : int
            The id of the lift to retrieve the start/end

        Returns
        -------
        tuple
            A tuple containing the start and end frame indices for the lift
            Note: NIOSH data is available in chunk_start and chunk_end+1 (the frame after the last frame of the lift)
        """
        chunk_start = self.CACHE_df[(self.CACHE_df['num_lifts'] == lift_id) & (self.CACHE_df['contact_state'] == 'carry')].iloc[0]["frame"]
        chunk_end = self.CACHE_df[(self.CACHE_df['num_lifts'] == lift_id) & (self.CACHE_df['contact_state'] == 'carry')].iloc[-1]["frame"]
        return chunk_start, chunk_end

    def select_lift(self):
        self.play = False
        currentLiftIndex = self.listWi_Lifts.currentRow()//2 + 1
        chunk_start, chunk_end = self.get_lifting_chunk(currentLiftIndex)
        self.hSlid_VideoMoverVon.setValue(chunk_start)
        self.hSlid_VideoMoverBis.setValue(chunk_end+1)

        self.frame_count = chunk_start if self.listWi_Lifts.currentRow() % 2 == 0 else chunk_end+1
        self.show_single_frame(self.frame_count)

    def speed_changed(self):
        self.lineE_FrameSpeed.setText(str(self.dial_frameSpeed.value()))
        self.skip_frames = self.dial_frameSpeed.value()

    def end_slider_changed(self):
        self.lab_frameBis.setText('bis ' + str(self.hSlid_VideoMoverBis.value()))
        if self.hSlid_VideoMoverVon.value() > self.hSlid_VideoMoverBis.value():
            self.hSlid_VideoMoverVon.setValue(self.hSlid_VideoMoverBis.value())
            self.lab_framVon.setText('von ' + str(self.hSlid_VideoMoverVon.value()))

    def start_slider_changed(self):
        self.lab_framVon.setText('von ' + str(self.hSlid_VideoMoverVon.value()))
        if self.hSlid_VideoMoverBis.value() < self.hSlid_VideoMoverVon.value():
            self.hSlid_VideoMoverBis.setValue(self.hSlid_VideoMoverVon.value())
            self.lab_frameBis.setText('bis ' + str(self.hSlid_VideoMoverBis.value()))

    def show_next_frame(self):
        self.play = False
        if self.frame_count != self.CACHE_df['frame'].iloc[-1]:
            self.frame_count = self.frame_count + 1
        self.show_single_frame(self.frame_count)

    def show_prev_frame(self):
        self.play = False
        if self.frame_count != self.CACHE_df['frame'].iloc[0]:
            self.frame_count = self.frame_count - 1
        self.show_single_frame(self.frame_count)

    def show_clip(self, start_idx, end_idx):
        self.play = True

        # use QTimer for scheduling the next frame update explicitly at known framerate
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        msec = int(1000/fps)
        self.frame_count = start_idx
        start_time = time.time()
        def play_next_frame():
            if VERBOSE:
                print(self.frame_count, end_idx, time.time()-start_time, self.playback_timer.timerId())
            self.show_single_frame(self.frame_count)
            if self.play and self.frame_count < end_idx:
                self.frame_count += 1
            else:
                self.playback_timer.stop()
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(play_next_frame)
        self.playback_timer.start(msec)

    def pause_video(self):
        if not self.play:
            self.play = True
            self.btn_pause.setText('Pause')
            self.show_clip(self.frame_count, self.hSlid_VideoMoverBis.value())
        else:
            self.btn_pause.setText('Weiter')
            self.play = False

    def stop_video(self):
        self.play = False

        # reset to start frame
        self.frame_count = self.hSlid_VideoMoverVon.value()
        self.show_single_frame(self.frame_count)

    def show_single_frame(self, idx):
        # update NIOSH data
        # num_lifts = max(1, self.CACHE_df.loc[idx]['num_lifts'])
        # chunk_start, chunk_end = self.get_lifting_chunk(num_lifts)
        # frame_data = self.CACHE_df.loc[chunk_end+1]  # always use last frame of current/least recent lift for NIOSH data (if idx is not 'carry' => use last frame of previous lift)
        frame_data = self.CACHE_df.loc[idx]
        self.set_lift_data(frame_data)

        # update image
        with utils.GetTime("update_image", verbose=VERBOSE):
            if not self.play:  # no playback means we have non-consecutive frames and need to seek to the correct frame
                # this incredibly slows down the GUI, so only use it for single frame selection and skip this for consecutive frames during playback TODO: this is due to how seeking in opencv works, use another library like PyAV?
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx - self.CACHE_df['frame'].iloc[0])
            received, image = self.get_image_from_stream()
            if received and image is not None:
                image = imutils.resize(image, width=self.img_width)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
                self.label.setPixmap(QPixmap.fromImage(image))

        # TODO: visualize currently displayed frame in the video sliders!
        self.lineE_LMM.setText(str(frame_data['pose_type']))
        self.lineE_idx.setText(str(idx))

        # refresh GUI
        QApplication.processEvents()

    def reset_pose_estimator(self):
        self.pose_estimator.reset()

    def next_frame(self):
        self.frame_count += self.skip_frames  # increase frame counter (first frame => idx=1)
        self.NIOSH.frame_count += self.skip_frames
        if self.skip_frames == 1:
            pass  # do nothing to avoid slow down => assert self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.frame_count-1
        else:
            if self.mode == Mode.CAPTURE:
                print("WARNING: Skipping frames in Mode.CAPTURE might cause bad contact detection!")
            # this incredibly slows down everything (even in case FORCE_TRAM=False and USE_HAND_CONTACT=False) TODO: this is due to how seeking in opencv works, use another library like PyAV?
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count-1)  # proceed to current frame (first frame => idx=0)
        image = self.image_processing()
        if image is None:
            print("Got an empty frame. Usually happens only once after the first loop (or if LOOP_VIDEO=False) when using local frames.")
            return False

        if not NO_GUI:
            if self.update:
                self.update_gui()

            # TODO: show this data live in gui for debugging purposes
            pose_type = PoseTypeLMM(self.CACHE[self.frame_count]['pose_type'])
            text = f"Current frame: {self.frame_count}\n"
            text += f"Total #Lifts:  {self.CACHE[self.frame_count]['num_lifts']}\n"
            text += f"Current contact state: {self.CACHE[self.frame_count]['detector_contact_state']}\n"
            text += f"Last contact score: {self.CACHE[self.frame_count]['detector_contact_score_last']:.3f}\n"
            text += f"Contact score: {self.CACHE[self.frame_count]['detector_contact_score']:.3f}\n"
            text += f"LMM classification:  {pose_type.name} [{pose_type.value}]\n"

            qimg = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qimg))
            QApplication.processEvents()
        self.writer_video.append_data(image)

        return True

    def get_image_from_stream(self):
        current_frame_id = self.cap.get(cv2.CAP_PROP_POS_FRAMES)  # position of current frame in video (zero based indexing)
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)     # total number of frames in video (not entirely reliable)
        current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)        # current position in a video file, measured in milliseconds
        fps = self.cap.get(cv2.CAP_PROP_FPS)                      # frames per second at which the video was captured
        codec = self.cap.get(cv2.CAP_PROP_FOURCC)                 # four-character code for the compression codec
        codec = bytearray.fromhex(hex(int(codec))[2:]).decode()
        if current_time > 0 and self.first_frame_time is None:
            self.first_frame_time = current_time
        if self.first_frame_time is not None:
            current_time = (current_time+self.first_frame_time)/1000   # frame_idx=0 & frame_idx=1 have current_time=0, frame_idx=2 has 1/fps => correct current_time for frame_idx>=2
        if VERBOSE:
            print(f"frame at {current_time:.3f}s {current_frame_id+1}/{total_frames} @{fps} ({codec})")  # => timing information from first to frames is unreliable (could better use current_frame_id/fps)
        return self.cap.read()

    def get_points_apple(self):
        try:
            # TODO: same with 3D Pose
            data = self.points_queue.get_nowait()
            joints = data['poses'][0]['joints']
            points2d = np.zeros([len(joints), 2], dtype=int)
            for joint, values in joints.items():
                points2d[JOINTS[self.skeleton_type].index(joint)] = [values['x']*self.img_width, (1-values['y'])*self.img_height]
            frame_id = data['frameNumber']
            #timestamp = data['timestamp']
            #print(f"Received json data for frame {frame_id} at timestamp {timestamp}")
            print(f"Received json data for frame {frame_id}")
            return points2d, points2d
        except queue.Empty:
            return None

    def draw_pose(self, image, points_2d):
        mask = np.logical_not(np.all(points_2d == 0, axis=1))

        n_bones = len(BONES[self.skeleton_type])
        n_joints = n_bones + 1

        assert points_2d.shape[0] == n_joints, points_2d.shape[1] == 2

        colors = get_bone_colors(self.skeleton_type)
        for i, b in enumerate(BONES[self.skeleton_type]):
            if b[0] in np.nonzero(mask)[0] and b[1] in np.nonzero(mask)[0]:
                image = cv2.line(image, points_2d[b[0]], points_2d[b[1]], color=COLORS[colors[i]], thickness=3)

        colors = get_joint_colors(self.skeleton_type)
        for i, j in enumerate(JOINTS[self.skeleton_type]):
            image = cv2.circle(image, points_2d[i], radius=5, color=COLORS[colors[i]], thickness=5)

    def image_processing(self):
        received, image = self.get_image_from_stream()
        if received is False or image is None:
            print("Invalid image retrieved from cv2.VideoCapture")
            # reset VideoCapture in case of frames loaded from file
            if self.frame_count % int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 1 and LOOP_VIDEO:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                received, image = self.get_image_from_stream()
                if received == False or image is None:
                    return None
            else:
                return None
        #image = np.ascontiguousarray(image[:, :, ::-1])  # inplace reordering is faster than using cv2.split() + cv2.merge(), but does not provide contiguous arrays
        # b, g, r = cv2.split(image)
        # image = cv2.merge((r, g, b))  # cv2.cvtColor is faster
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # get human pose
        pts_3d_hq = None
        pts_2d_hq = None
        # in case of webcam we need TRAM to get the pose as no apple detections are available
        if FORCE_TRAM or self.is_webcam:
            print("WARNING: LMM pose classification is not optimized for tram detections!")
            with utils.GetTime("tram", verbose=VERBOSE):
                pts_2d, pts_3d_ = self.pose_estimator.predict(image)  # takes approx. 0.12 seconds => max. 8 FPS (bottleneck: smpl_head transformer)
            pts_3d = utils.normalize_pose(pts_3d_, self.skeleton_type) if pts_3d_ is not None else None
            if pts_2d is not None:
                self.person_detected_count += 1
            else:
                self.person_detected_count = 0
                self.contact_detector.last_load_centers.clear()
            self.person_detected = self.person_detected_count > 3
            pts_3d_hq = pts_3d
            pts_2d_hq = pts_2d
        else:
            with utils.GetTime("Retrieving Pose", verbose=VERBOSE):
                if self.cap_src == 'stream':
                    pts_2d, pts_3d_ = self.get_points_apple()  # live data from iPhone => raw apple data
                else:
                    pts_2d, pts_3d_ = self.pose_loader.load(image, frame_idx=int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))-1)  # -1 as frame position already increased
                    pts_2d_hq, pts_3d_hq_ = self.pose_loader_hq.load(image, frame_idx=int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))-1)  # -1 as frame position already increased
                    pts_3d_hq = utils.normalize_pose(pts_3d_hq_, self.skeleton_type) if pts_3d_hq_ is not None else None
                pts_3d = utils.normalize_pose(pts_3d_, self.skeleton_type) if pts_3d_ is not None else None
                self.person_detected = pts_2d is not None

        with utils.GetTime("Computing Contact", verbose=VERBOSE):
            # predict contact and compute metrics
            contact_state_change = False
            if USE_HAND_CONTACT:
                # predict contact state change at the current frame using the hand contact detector
                pts2d = pts_2d_hq if pts_2d_hq is not None and self.cap_src != 'stream' else pts_2d  # in case of live stream we use apple detections (tram only available for frames of interest)
                pts3d = pts_3d_hq if pts_3d_hq is not None and self.cap_src != 'stream' else pts_3d
                contact_state_change_pred = self.contact_detector.predict(image, pts3d, pts2d)
                # update postprocessing buffer with information for current frame
                self.contact_postprocessor.last_state_changes.append(contact_state_change_pred)
                # compute the latest index of contact state change based on the current buffers/sliding windows
                contact_state_change, contact_change_idx, last_load_dist, rtg, contact_state_change_filtered, root_d = self.contact_postprocessor.check_contactStateChange(current_frame=self.frame_count, pts_3d_=pts_3d_)

                if contact_state_change == 'drop' and self.NIOSH.V_start is None:  # no preceding lift detected -> sequence started with carry => ignore change
                    contact_state_change = False

                self.update = False
                if not FORCE_TRAM and not self.is_webcam and self.cap_src == 'stream':  # live data from iPhone => we need to compute tram only for frames of interest
                    assert pts_3d_hq is None
                    with utils.GetTime("Caching Input", verbose=VERBOSE):
                        if pts_2d is not None:
                            self.pose_estimator.last_box = np.asarray([pts_2d.min(axis=0)[0], pts_2d.min(axis=0)[1],
                                                                       pts_2d.max(axis=0)[0], pts_2d.max(axis=0)[1], 1.0], dtype=np.float32)
                        pts_2d_hq, pts_3d_hq_ = self.pose_estimator.predict(image, cache_input_only=True)  # only cache image data for later use
                        # TODO: caching images in PoseEstimator needs to be extended to contain more than the last 16 images as contact may occur earlier
                if contact_state_change:
                    print(f"Change of contact state detected: {contact_state_change}")
                    if contact_state_change == 'lift':
                        self.num_lifts += 1

                    # evaluate NIOSH on estimated contact change frame
                    if pts_3d_hq is None:
                        # TODO: caching images in PoseEstimator needs to be extended to contain more than the last 16 images as contact may occur earlier
                        print("WARNING: NIOSH requires 3D points at 'contact_change_idx' for evaluation! Here we compute at the current frame only!")
                        pts_2d_hq, pts_3d_hq_ = self.pose_estimator.predict(image, compute_only=True)  # only compute poses (image has already been queued)
                        pts_3d_hq = utils.normalize_pose(pts_3d_hq_, self.skeleton_type) if pts_3d_hq_ is not None else None
                    # it might happen that the current frame is selected => add pts_3d to CACHE here, otherwise NIOSH computation would crash
                    if self.frame_count == contact_change_idx:
                        self.CACHE[contact_change_idx] = {'points3d_hq': pts_3d_hq, 'contact_state': self.CACHE[max(self.CACHE.keys())]['contact_state']}
                    if contact_state_change == 'lift':
                        multipliers, scores = self.NIOSH.start_lifting(self.CACHE[contact_change_idx]['points3d_hq'])
                    else:
                        multipliers, scores = self.NIOSH.compute_after_lift(self.CACHE[contact_change_idx]['points3d_hq'])
                    self.update = True
                    # TODO: check if lift/drop frames are too close (< 10 frames is most likely are wrong detection...) (Aufnahme 8)
        # from skeletons import plot
        # plot(pts_3d_, '2D')

        # cache results/poses
        self.CACHE[self.frame_count] = {
            'frame': self.frame_count,
            'points3d': pts_3d,
            'points2d': pts_2d,
            'points3d_hq': pts_3d_hq,
            'points2d_hq': pts_2d_hq,
            'person_detected': self.person_detected,
            'computed_contact_state_change': contact_state_change,
            'computed_contact_state_change_at': contact_change_idx if contact_state_change else -1,
            'detector_contact_state_change': contact_state_change_pred if USE_HAND_CONTACT else False,
            'detector_contact_state_change_filtered': contact_state_change_filtered if USE_HAND_CONTACT else False,
            'detector_contact_state': self.contact_detector.contact_state if USE_HAND_CONTACT else 'none',
            'detector_contact_score_last': self.contact_detector.last_scores[-1] if USE_HAND_CONTACT else 0,
            'detector_contact_score': np.mean(self.contact_detector.last_scores) if USE_HAND_CONTACT else 0,

            # is overwritten when next contact_change_idx is computed:
            'num_lifts': self.num_lifts if USE_HAND_CONTACT else 0,
            'contact_state_change': False,
            'contact_state': self.CACHE[max(self.CACHE.keys())]['contact_state'] if len(self.CACHE) else 'none',
            
            # logging for contact postprocessing
            'last_load_dist': last_load_dist,
            'rtg': rtg,
            'root_dx': root_d['root_dx'],
            'root_dy': root_d['root_dy'],
            'root_dz': root_d['root_dz'],
            'root_x': root_d['root_x'],
            'root_y': root_d['root_y'],
            'root_z': root_d['root_z'],
            'root_dxz': root_d['root_dxz'],
        }
        if contact_state_change:
            # log NIOSH values at contact_change_idx (== frame for which they are evaluated)
            for name, value in multipliers.items():  # flatten all dictionary data for compact pandas view
                self.CACHE[contact_change_idx][f'NIOSH_{name}'] = value
            for name, value in scores.items():
                self.CACHE[contact_change_idx][f'NIOSH_{name}'] = value
            self.CACHE[contact_change_idx]['contact_state_change'] = contact_state_change
            for idx in range(contact_change_idx, self.frame_count+1):
                self.CACHE[idx]['contact_state'] = 'carry' if contact_state_change == 'lift' else 'none'
                self.CACHE[idx]['num_lifts'] = self.num_lifts

        # classify pose for LMM
        if not (FORCE_TRAM or self.is_webcam):
            # raw apple data required for LMM pose classification
            if self.cap_src == 'stream':
                pts_3d_raw = pts_3d_
            else:
                pts_3d_raw = self.pose_loader.points3d[int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))-1]  # -1 as frame position already increased]
            pose_type, pose_type_raw = self.pose_classificator.predict(pts_3d_raw)
            if VERBOSE:
                print(self.pose_classificator.count, pose_type.name, pose_type.name)
        else:
            pose_type = PoseTypeLMM.NONE
            pose_type_raw = PoseTypeLMM.NONE
        self.CACHE[self.frame_count]['pose_type'] = pose_type.value
        self.CACHE[self.frame_count]['pose_type_raw'] = pose_type_raw.value

        # draw image
        if self.person_detected:
            pts = pts_2d_hq if pts_2d_hq is not None and self.cap_src != 'stream' else pts_2d  # in case of live stream we use apple detections (tram only available for frames of interest)
            self.draw_pose(image, pts)
        image = cv2.resize(image, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        return image

    def start_thread(self):
        self.thread = threading.Thread(target=self.update_thread)
        self.thread_run = True
        self.thread.start()

    def update_thread(self):
        while self.thread_run:

            if self.mode == Mode.CAPTURE:
                self.thread_run = self.next_frame() or LOOP_VIDEO

            elif self.mode == Mode.REPLAY:
                if self.CACHE_df is None:
                    # TODO: move to switch_mode()?
                    # just stopped capturing or requested change to replay mode (load data from RESULT_DIR)
                    # => stop the video, load it using VideoCapture and convert/load CACHE to dataframe
                    if len(self.CACHE):
                        df = pd.DataFrame.from_dict(self.CACHE, orient='index')
                        df.to_pickle(Path(RESULT_DIR, 'gui_data.pkl'))
                    else:
                        df = pd.read_pickle(Path(RESULT_DIR, 'gui_data.pkl'))

                    self.writer_video.close()
                    self.cap.release()
                    self.cap = cv2.VideoCapture(str(Path(RESULT_DIR, 'video_2d.mp4')))
                    self.load_dataframe(df)

            if RECORD_GUI:
                with mss.mss() as sct:
                    monitor = {
                        "top": self.geometry().y(),
                        "left": self.geometry().x(),
                        "width": self.geometry().width(),
                        "height": self.geometry().height()
                    }
                    if self.writer_gui is None:
                        fps = self.cap.get(cv2.CAP_PROP_FPS)
                        self.writer_gui = imageio.get_writer(Path(RESULT_DIR, 'gui_capture.mp4'), fps=fps, **dict(codec='h264', ffmpeg_params=['-crf', '17'], macro_block_size=None))
                    else:
                        # first frame has weird shape -> do not capture
                        scr_shot = sct.grab(monitor)  # not always grabbing the most recent frame when using VNC
                        img = np.asarray(Image.frombytes("RGB", scr_shot.size, scr_shot.bgra, "raw", "BGRX"))

                        # ensure width and height are even, otherwise libx264 will crash
                        height, width, _ = img.shape
                        img = img[:height - (height % 2), :width - (width % 2), :]

                        self.writer_gui.append_data(img)
        print("Quit")
        QApplication.quit()  # quit event loop (app.exec_())

    def start_server(self):
        self.server_thread = threading.Thread(target=self.run_server)
        self.server_thread_run = True
        # Server needs to run in a daemon thread, otherwise it will block the main thread from closing
        # => The entire Python program exits when no alive non-daemon threads are left (https://docs.python.org/3/library/threading.html#threading.Thread.daemon)
        self.server_thread.daemon = True
        self.server_thread.start()

    def run_server(self):
        self.server.run(debug=True, use_reloader=False, host="0.0.0.0")
        # host param required to run flask on all IP adresses (otherwise it runs only on localhost and it is not reachable externally)
        # https://flask.palletsprojects.com/en/3.0.x/quickstart/#public-server

    def setup_flask_routes(self):
        @self.server.route('/points', methods=['POST'])
        def update_points():
            #print(request.data[:100]) # => sometimes we get empty data due to not fully implemented 3D skeleton posts (converting to json fails and triggers automatic ACK as "POST /points HTTP/1.1" 415 -)
            data = request.get_json()
            if data:
                self.points_queue.put(data)
            return jsonify(success=True), 200

    def main(self, app):
        self.start_server()  # handling of incoming skeleton data
        self.start_thread()  # handling of incoming video frames

        self.show()  # show the window
        app.exec_()  # execute event loop
        self.thread_run = False
        self.server_thread_run = False
        time.sleep(0.5)
        self.thread.join(timeout=0.5)
        self.server_thread.join(timeout=0.5)


if __name__ == '__main__':
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication.instance()  # https://stackoverflow.com/questions/46304070/multiple-qapplication-instances
    if app is None:
        print("No QApplication instance found -> no other qt-based visualizer has been loaded")
        app = QApplication(sys.argv)
    gui = AkevapVisual()

    try:
        if NO_GUI:
            while True:
                if not gui.next_frame():
                    break
        else:
            gui.main(app)
    except Exception as e:
        print(e)
    finally:
        # import pandas as pd
        # df = pd.DataFrame.from_dict(gui.CACHE, orient='index')
        # df.to_pickle(Path(RESULT_DIR, 'gui_data.pkl'))

        if gui.cap.isOpened():
            gui.cap.release()
        if RECORD_GUI:
            gui.writer_gui.close()
        gui.writer_video.close()
        del gui
