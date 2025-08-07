import numpy as np
from utils_parse import parse_alphapose3D_data, parse_ios3D_data_new
from pathlib import Path
import json
from scipy.spatial.transform import Rotation

class PoseGTLoader:

    def __init__(self, DATA_DIR, initial_frame_idx, detector3D='apple'):
        self.count = initial_frame_idx
        self.use_detector3D = detector3D

        if 'data_23-11-15' in str(DATA_DIR) or 'data_ebeling' in str(DATA_DIR) or 'Public/tmp/' in str(DATA_DIR):
            self.data_format = 'old'
        elif 'data_24-04-30' in str(DATA_DIR):
            self.data_format = 'new'
        else:
            self.data_format = 'new'

        if self.use_detector3D == 'tram':
            with open(Path(DATA_DIR, "tram/tram-results.json"), 'r') as f:
                self.data3D = json.load(f)
            self.points3d = parse_alphapose3D_data(self.data3D, trial=Path(DATA_DIR).name)
        elif self.use_detector3D == 'apple':
            if self.data_format == 'old':
                with open(Path(DATA_DIR, "3D_Keypoints_fixed.json"), 'r') as f:  # "3D_Keypoints_fixed.json" for old recordings
                    data = f.read()
            else:
                with open(Path(DATA_DIR, "3D_Keypoints.json"), 'r') as f:  # "3D_Keypoints.json" for newer recordings
                    data = f.read()
            self.data3D = json.loads(data)['all3DPoses']
            self.points3d = parse_ios3D_data_new(self.data3D, trial=Path(DATA_DIR).name)  # json format is identical for old and new recordings (only filenames and intrinsics differ)
        else:
            print(f"Failed to load ground truth data! {self.use_detector3D}")
            exit()

        # Indexing with the following mapping results in a compacted joint list in the order of the target skeleton format,
        # where joints without any correspondence are missing.
        # Therefore, we need to select the joints of interest before assigning (subset of joints for which we have correspondences)
        self.smpl2apple = [15, 12, 17, 19, 21, 16, 18, 20,  0,  2,  5,  8,  1,  4,  7]  # smpl.map_to(openpose)  # iOS_2D is a subset of OpenPose, so we can use that mapping
        self.joints_with_correspondence_for_smpl = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]  # openpose.joints_with_correspondences(smpl)
        self.h36m2apple = [ 9,  8, 14, 15, 16, 11, 12, 13,  0,  1,  2,  3,  4,  5,  6]  # H36m.map_to(openpose)  # iOS_2D is a subset of OpenPose, iOS_3D is similar to H3.6m
        self.joints_with_correspondence_for_h36m = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]  # openpose.joints_with_correspondences(H36m)

    def load(self, image, frame_idx):
        height, width = image.shape[0], image.shape[1]
        if self.data_format == 'old':
            img_focal = 1100  # old tram offline detections were computed using f=1100 (not using the original focal length)
        else:
            img_focal = 1400  # new tram offline detections were computed using f=1400 (close to the original focal length), for apple the original focal length is used (see below)
        img_center = [width/2, height/2]
        intrinsics = self.build_intrinsics(dict(fx=img_focal, fy=img_focal, cx=img_center[0], cy=img_center[1]))

        if self.count % len(self.points3d) != frame_idx:
            if self.use_detector3D == 'apple':
                # might fail with apple detections, as they are not guaranteed to exist for every frame => add empty frame to force reusing last frame
                self.points3d = np.insert(self.points3d, frame_idx, 0, axis=0)
            else:
                assert False  # should not happen with tram detections

        # if empty data: reuse old data
        while np.allclose(self.points3d[frame_idx], 0):
            print(f"Ground truth missing for frame {frame_idx}, reusing last frame")
            frame_idx -= 1
        pred_j3d_w = self.points3d[frame_idx]  # 24 x 3
        self.count += 1

        # points provided in "3D_Keypoints.json" are in skeleton coordinate system => transform to camera coords
        if self.use_detector3D == 'apple':
            if self.data_format == 'new':
                # reading intrinsics from json is only possible with "3D_Keypoints.json" for newer recordings! (if no intrinsics are known, Apple assumes f=1100 and cx/cy = 0.5 * w/h)
                # TODO: use known intrinsics also for new tram-results.json
                intrinsics = self.data3D[frame_idx]['poses'][0]['cameraIntrinsicMatrix']
                img_height = self.data3D[frame_idx]['poses'][0]['intrinsicMatrixRefDimHeight']
                img_width = self.data3D[frame_idx]['poses'][0]['intrinsicMatrixRefDimWidth']
                intrinsics = self.build_intrinsics(dict(fx=intrinsics['row1']['x'], fy=intrinsics['row2']['y'], cx=img_width-intrinsics['row1']['z'], cy=img_height-intrinsics['row2']['z']))

            # load extrinsics
            c = self.data3D[frame_idx]['poses'][0]['cameraOriginMatrix']
            extrinsics = np.asarray([[c['row1']['x'], c['row1']['y'], c['row1']['z'], c['row1']['w']],
                                     [c['row2']['x'], c['row2']['y'], c['row2']['z'], c['row2']['w']],
                                     [c['row3']['x'], c['row3']['y'], c['row3']['z'], c['row3']['w']],
                                     [c['row4']['x'], c['row4']['y'], c['row4']['z'], c['row4']['w']]])

            # fix extrinsics for data recorded from iphone
            # SceneKit uses a right-handed coordinate system where (by default) the direction of view is along the negative z-axis, as illustrated below. [https://developer.apple.com/documentation/scenekit/organizing_a_scene_with_nodes]
            # => rotate camera 180° around y-axis as apple lokks in neg z-direction
            y180 = Rotation.from_euler('y', 180, degrees=True).as_matrix()
            extrinsics[:3, :3] = y180 @ extrinsics[:3, :3]
            extrinsics[:3, 3] = y180 @ extrinsics[:3, 3]

            # transform 3D points to camera coordinate system
            pred_j3d_w = (extrinsics @ np.concatenate([pred_j3d_w, np.ones_like(pred_j3d_w[:, [0]])], axis=1).T).T  # N x 4
            pred_j3d_w = pred_j3d_w[:, :3]  # N x 3

            # points3d from tram/smpl are provided in camera coordinates: camera is fixed at the origin but y axis is pointing down and x to the right
            # => rotation around z axis required
            z180 = Rotation.from_euler('z', 180, degrees=True).as_matrix()
            pred_j3d_w = (z180 @ pred_j3d_w.T).T  # => rotate the camera coordinate system (misalignment between camera and world coordinate system by rotation around z axis)

        # project to 2D
        pred_j2d_proj = (intrinsics @ pred_j3d_w.T).T  # 24 x 3
        pred_j2d_proj = pred_j2d_proj[:, :2] / pred_j2d_proj[:, [2]]
        pred_j2d_proj = pred_j2d_proj[:, :2]

        if self.use_detector3D == 'tram':
            pts = np.zeros([19, 2], dtype=np.float32)  # empty pose for apple's 2D skeleton
            pts[self.joints_with_correspondence_for_smpl] = pred_j2d_proj[self.smpl2apple, :]
            pts_3d = np.zeros([19, 3], dtype=np.float32)
            pts_3d[self.joints_with_correspondence_for_smpl] = pred_j3d_w[self.smpl2apple, :]
        else:
            pts = np.zeros([19, 2], dtype=np.float32)
            pts[self.joints_with_correspondence_for_h36m] = pred_j2d_proj[self.h36m2apple, :]
            pts_3d = np.zeros([19, 3], dtype=np.float32)
            pts_3d[self.joints_with_correspondence_for_h36m] = pred_j3d_w[self.h36m2apple, :]

        return pts.astype(int), pts_3d

    def build_intrinsics(self, intr):
        mat = np.zeros([3, 3])
        mat[0, 0] = intr['fx']
        mat[1, 1] = intr['fy']
        mat[0, 2] = intr['cx']
        mat[1, 2] = intr['cy']
        mat[2, 2] = 1
        return mat
