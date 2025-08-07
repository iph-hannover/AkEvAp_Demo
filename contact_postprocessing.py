import numpy as np
import utils
from collections import deque
import pandas as pd
import more_itertools
from scipy.signal import argrelextrema, savgol_filter

class ContactPostprocessor:

    def __init__(self, contact_detector, CACHE):
        self.last_state_changes = deque(maxlen=30)
        self.contact_detector = contact_detector
        self.CACHE = CACHE

        # Buffer for contact_state
        self.contact_state_buffer = deque(maxlen=40)
        self.contact_state_window_size = 8  # no two contact state is allowed simultaneously on each frame window
        self.last_detector_contact_states = deque(maxlen=30)
        self.last_load_dists = deque(maxlen=30)
        self.doubleDetection = False

        # Buffer for occlusion
        self.orientation_buffer = deque(maxlen=30)
        self.occlusion_detected = False
        self.load_occlusion_buffer = deque(maxlen=30)
        self.load_occlusion_upperbody_buffer = deque(maxlen=30)
        self.occlusion_window_size = 24
        self.occlusion_autostopper = True
        self.occlusion_autostop = False
        self.occlusion_autostopper_window = 30
        self.frames_since_last_occlusion = 0
        self.last_going_false_index = None
        self.joints = utils.get_joint_indices(skeleton_type="2D")
        self.root_x = deque([0], maxlen=30)
        self.root_y = deque([0], maxlen=30)
        self.root_z = deque([0], maxlen=30)
        self.dxz = deque(maxlen=30)
        self.use_root_speed_for_extrema = True

    def check_contactStateChange(self, current_frame, pts_3d_):
        contact_state_change = False
        contact_change_idx = -1

        # search for local load_distance maxima close to a state change and evaluate NIOSH on that max-frame (contact detection is delayed due to smoothing -> assume that contact change occurs at the most distant arm stretch)
        last_state_changes = list(self.last_state_changes)

        ## Smoothen last_state_changes again
        # Create strides / sliding windows
        windows = list(more_itertools.windowed(last_state_changes, self.contact_state_window_size, fillvalue=False))
        # Only delete every simultaneous occurence of two state changes in a window
        for w_idx, w in enumerate(windows):
            # Indexes of the two state changes in a window --> to be removed
            tmp = list([idx for idx,i in enumerate(w) if i != False])
            if len(tmp) >= 2:
                # replace the first two with Falses
                self.last_state_changes[w_idx+tmp[0]] = False
                self.last_state_changes[w_idx+tmp[1]] = False
                # correct the cache
                # print(tmp)
                self.CACHE[current_frame-(self.contact_state_window_size-tmp[0])+1]["detector_contact_state_change_filtered"] = False
                break

        # add to detector contact states buffer
        detector_contact_state_ = self.contact_detector.contact_state
        self.last_detector_contact_states.append(detector_contact_state_)

        # distance for logging
        last_load_centers = np.asarray(self.contact_detector.last_load_centers)[-1]
        last_load_dist = np.linalg.norm(last_load_centers)
        self.last_load_dists.append(last_load_dist)

        # first, append "False" to contact state buffer
        self.contact_state_buffer.append(False)

        ## Get body orientation, load center orientation, and check occlusion
        rtg = self.get_body_orientation(pts_3d_) # richtung, angle_hip, richtung_binary
        self.orientation_buffer.append(rtg[0])
        root_dx = pts_3d_[self.joints.ROOT][0] - self.root_x[-1]
        root_dy = pts_3d_[self.joints.ROOT][1] - self.root_y[-1]
        root_dz = pts_3d_[self.joints.ROOT][2] - self.root_z[-1]
        self.dxz.append(np.sqrt(root_dx*root_dx+root_dz*root_dz))
        self.root_x.append(pts_3d_[self.joints.ROOT][0])
        self.root_y.append(pts_3d_[self.joints.ROOT][1])
        self.root_z.append(pts_3d_[self.joints.ROOT][2])
        root_d = {
            'root_dx': root_dx,
            'root_dy': root_dy,
            'root_dz': root_dz,
            'root_x': self.root_x-root_dx,
            'root_y': self.root_y-root_dy,
            'root_z': self.root_z-root_dz,
            'root_dxz': self.dxz[-1],
        }

        ## Occlusion Autostopper
        contact_state_change, last_state_changes = self.occlusion_autostopper_method(rtg, contact_state_change, last_state_changes, current_frame)

        ## A contact change
        # Conditions
        # 1) 20 frames have passed
        # 2) Theres something in last_state_changes
        # 3) More than 7 "carry" in the detector_contact_change
        if len(last_state_changes) >= 20 and any(x is not False for x in last_state_changes) and len([x for x in self.last_detector_contact_states if x is not False]) >= 7:
            last_state_change_idx = [x is not False for x in last_state_changes].index(True)

            # Search for a local maxima of the load distance that is nearest to the last_state_change_index
            # Get load_distances
            load_dists = np.asarray(self.last_load_dists)

            # Get maxima indexes
            load_dist_maxima_indexes = argrelextrema(load_dists, np.greater, mode='wrap')[0]
            # print("load_dists: ", load_dists)
            # print("load_dist_maxima_indexes: ", load_dist_maxima_indexes)

            ## Method for correlating frame (which frame shall the state change to be assigned to)
            # 1) use minimal root_speed (minimum root xz translation)
            if self.use_root_speed_for_extrema:
                dxz = np.squeeze(np.array(self.dxz))
                # smoothen
                dxz = savgol_filter(np.squeeze(dxz), min(15, len(dxz)), 3)
                dxz_at_maximas = list(dxz[load_dist_maxima_indexes])
                min_dxz_index = np.argmin(dxz_at_maximas)
                load_dist_max_idx = load_dist_maxima_indexes[min_dxz_index]
                load_dist_max = load_dists[load_dist_max_idx]

            # 2) using nearest maxima of load distance
            else:
                # Look for index which are nearest to last_state_change_idx
                nearest_maxima = np.argmin(np.abs(np.array(load_dist_maxima_indexes)-last_state_change_idx))
                load_dist_max_idx = load_dist_maxima_indexes[nearest_maxima]
                # load_dist "used" for the state change (actually not the absolute max, it is the nearest maxima instead)
                load_dist_max = load_dists[load_dist_max_idx]

            # once +/-15 frames are computed -> search for max load distance
            if last_state_change_idx == 20 or self.occlusion_autostop:  # look -20 to +10 frame, 15 look +/-15 frame around
                self.occlusion_autostop = False
                contact_frame_idx = current_frame - len(last_state_changes) + load_dist_max_idx + 1  # next frame the state change would be dropped, so we assume change at max distance; +1 => frame counting starts with 1
                contact_change_idx = contact_frame_idx
                if self.use_root_speed_for_extrema:
                    print(f"Hands stretched: {load_dist_max*100:.1f} with minimum root movement {dxz[load_dist_max_idx]}cm/frame. (frame={load_dist_max_idx}), contact_frame_idx={contact_frame_idx}")
                else:
                    print(f"Hands stretched at max: {load_dist_max*100:.1f} cm (frame={load_dist_max_idx}), contact_frame_idx={contact_frame_idx}")
                contact_state_change = last_state_changes[last_state_change_idx]

                ## Detect body occlusion
                print("contact_state_change before: ", contact_state_change)
                contact_state_change = self.detect_occlusion(contact_state_change, last_state_changes, last_state_change_idx, load_dist_max_idx, contact_frame_idx, current_frame)
                print("contact_state_change after: ", contact_state_change)

                if contact_state_change is not False:
                    ## Modify contact_state_change using buffer
                    # Index: at the index of load_dist + preceding elements in contact_state_buffer
                    contact_state_buffer_idx = load_dist_max_idx + (len(self.contact_state_buffer)-len(last_state_changes))
                    # Cancel contact state change if it's position in the buffer already assigned (not False)
                    if self.contact_state_buffer[contact_state_buffer_idx]:
                        # Set as False
                        self.contact_state_buffer[contact_state_buffer_idx] = False
                        contact_state_change = False
                        print("Double detection at "+str(contact_frame_idx)+", contact state change deleted.")
                        self.doubleDetection = True

                    # At this point it is a truly new contact_state_change, then...
                    # Add contact_state_change to buffer first
                    del self.contact_state_buffer[contact_state_buffer_idx]
                    self.contact_state_buffer.insert(contact_state_buffer_idx, contact_state_change)

                    # To ensure the "used" local maxima will not be used again to determine next state change,
                    # replace local load_dists with Zeros
                    # so that the next detection automatically uses next available maxima
                    load_dist_minimas = argrelextrema(load_dists, np.less, mode='wrap')[0] # Index of the minimas

                    minimas_after_load_dist_max_idx = np.argwhere(load_dist_minimas > load_dist_max_idx)
                    # print("LLD b4:", self.last_load_dists)
                    # If there's any next minima
                    if len(minimas_after_load_dist_max_idx) > 0:
                        # replace self.last_load_dists until the next minima, so that double detection on a same maxima can't occur
                        print(load_dist_minimas[minimas_after_load_dist_max_idx[0][0]])
                        for _ in range(0, load_dist_minimas[minimas_after_load_dist_max_idx[0][0]]):
                            self.last_load_dists[_] = 0.0
                    # Otherwise, delete until this maxima's index
                    else:
                        # for _ in range(0, load_dist_max_idx+1): # +1 including deleting this maxima --> causing tiny blocks!
                        for _ in range(0, load_dist_max_idx):
                            self.last_load_dists[_] = 0.0
                        # = delete the entire deque
                        # self.last_load_dists.clear()
                    # print("LLD:", self.last_load_dists)
                self.last_state_changes[last_state_change_idx] = False  # mark state change as processed
        return contact_state_change, contact_change_idx, last_load_dist, rtg, last_state_changes[-1], root_d

    def get_body_orientation(self, keypoints3D):
        # keypoints3D: origin is camera coordinate

        # get body angle from R-Hip - L-Hip, seen from top/y-axis. Take only the x & z coordinate
        l_hip = np.array([keypoints3D[self.joints.L_HIP][0], keypoints3D[self.joints.L_HIP][2]])
        r_hip = np.array([keypoints3D[self.joints.R_HIP][0], keypoints3D[self.joints.R_HIP][2]])
        deltaZ = l_hip[1] - r_hip[1]
        deltaX = l_hip[0] - r_hip[0]
        angleInDegrees = np.degrees(np.arctan2(deltaZ, deltaX)) # 0 faces camera, 90 faces right 180 faces back, -90 faces left

        # if abs(angleInDegrees) >= 115:
        if abs(angleInDegrees) >= 135: #125
            richtung = "back"
            richtung_binary = True
        else:
            richtung = "front"
            richtung_binary = False

        return richtung, angleInDegrees, richtung_binary

    def get_load_center_orientation(self, keypoints3D):
        # keypoints3D = utils.normalize_pose(keypoints3D, self.skeleton_type) if keypoints3D is not None else None
        load_center = utils.compute_load_center(keypoints3D, self.skeleton_type)
        load_center = np.asarray([load_center[0], load_center[2]])
        root  = np.array([keypoints3D[self.joints.ROOT][0], keypoints3D[self.joints.ROOT][2]])
        deltaZ = load_center[1] - root[1]
        deltaX = load_center[0] - root[0]
        angleInDegrees = np.degrees(np.arctan2(deltaZ, deltaX)) # 0 faces front, 90 faces right 180 faces back, -90 faces left
        dist_2D = np.linalg.norm([deltaZ, deltaX])
        return angleInDegrees, dist_2D

    def check_occlusion_using_load_to_pelvis_position(self, pts_3d_, mode="TRAM"):
        # Possible occlusion: if the load center seen from camera not in between l_hip and r_hip +- 0,5 hip_width
        # --> the x coordinate
        keypoints3D = pts_3d_
        load_center = utils.compute_load_center(keypoints3D, self.skeleton_type)
        load_center = np.asarray([load_center[0], load_center[2]])
        l_hip = np.array([keypoints3D[self.joints.L_HIP][0], keypoints3D[self.joints.L_HIP][2]])
        r_hip = np.array([keypoints3D[self.joints.R_HIP][0], keypoints3D[self.joints.R_HIP][2]])

        hip_width = abs(r_hip[0] - l_hip[0])

        # Only if l_hip < r_hip
        if l_hip[0] < r_hip[0]:
            if load_center[0] > (l_hip[0] - 0.5*hip_width) and load_center[0] < (r_hip[0] + 0.5*hip_width):
                print("OCC")
                return True
            else:
                return False
        else:
            return False

    def check_occlusion_using_load_to_upperbody_position(self, pts_3d_, mode="TRAM"):
        # Possible occlusion: if the load center seen from camera not in between l_hip and r_hip +- 0,5 hip_width
        # --> the x coordinate
        keypoints3D = pts_3d_
        load_center = utils.compute_load_center(keypoints3D, self.skeleton_type)
        load_center = np.asarray([load_center[0], load_center[2]])
        l_hip = np.array([keypoints3D[self.joints.L_HIP][0], keypoints3D[self.joints.L_HIP][2]])
        r_hip = np.array([keypoints3D[self.joints.R_HIP][0], keypoints3D[self.joints.R_HIP][2]])
        l_elbow = np.array([keypoints3D[self.joints.L_ELBOW][0], keypoints3D[self.joints.L_ELBOW][2]])
        r_elbow = np.array([keypoints3D[self.joints.R_ELBOW][0], keypoints3D[self.joints.R_ELBOW][2]])
        l_shoulder = np.array([keypoints3D[self.joints.L_SHOULDER][0], keypoints3D[self.joints.L_SHOULDER][2]])
        r_shoulder = np.array([keypoints3D[self.joints.R_SHOULDER][0], keypoints3D[self.joints.R_SHOULDER][2]])

        hip_width = abs(r_hip[0] - l_hip[0])

        most_left = np.min([l_hip[0], l_elbow[0], l_shoulder[0]])
        most_right = np.max([r_hip[0], r_elbow[0], r_shoulder[0]])
        most_left_arg = np.argmin([l_hip[0], l_elbow[0], l_shoulder[0]])
        most_right_arg = np.argmax([r_hip[0], r_elbow[0], r_shoulder[0]])

        l, r = None, None
        # Only if l_hip < r_hip
        if l_hip[0] < r_hip[0]:
            if load_center[0] > (most_left - 0.5*hip_width) and load_center[0] < (most_right + 0.5*hip_width):
                match(most_left_arg):
                    case 0:
                        l = " l_hip"
                    case 1:
                        l = " l_elbow"
                    case 2:
                        l = " l_shoulder"
                match(most_right_arg):
                    case 0:
                        r = " r_hip"
                    case 1:
                        r = " r_elbow"
                    case 2:
                        r = " r_shoulder"
                print("OCC_upperbody by "+l+r)
                return True
            else:
                return False
        else:
            return False

    def occlusion_autostopper_method(self, rtg, contact_state_change, last_state_changes, current_frame):
        # If it is an occlusion (a drop already deleted), but body orientation occlusion is False
        if self.occlusion_detected and not rtg[2]: # self.orientation_buffer[-1]
            self.frames_since_last_occlusion = self.frames_since_last_occlusion + 1
            print("AUTOSTOP", self.frames_since_last_occlusion)
            if self.occlusion_autostopper and self.frames_since_last_occlusion > self.occlusion_autostopper_window:
                # add drop at index self.occlusion_autostopper_window/2
                contact_state_change = "drop"
                self.last_state_changes[-int(self.occlusion_autostopper_window/2)] = "drop"
                # change detector contact states to carry
                for i in range(int(self.occlusion_autostopper_window/2)):
                    self.last_detector_contact_states[-i] = "carry"
                # reassign last_state_changes
                last_state_changes = list(self.last_state_changes)
                # change flags
                self.occlusion_detected = False
                self.frames_since_last_occlusion = 0
                self.occlusion_autostop = True
                print("OCCLUSION AUTOSTOPPER: occlusion stopped to frame ", current_frame-int(self.occlusion_autostopper_window/2))
        return contact_state_change, last_state_changes

    def detect_occlusion(self, contact_state_change, last_state_changes, last_state_change_idx, load_dist_max_idx, contact_frame_idx, current_frame):
        ## Detect body occlusion here
        # possible_occlusion: check if theres a switch from front (False) to back (True)
        temp_df = pd.DataFrame()
        # Only add max: +- self.occlusion_window_size elements around
        temp_df["orientation_buffer"] = list(self.orientation_buffer)[max(load_dist_max_idx-int(self.occlusion_window_size/2), 0):min(load_dist_max_idx+int(self.occlusion_window_size/2), len(self.orientation_buffer))]
        temp_df.index = np.array(range(len(temp_df["orientation_buffer"]))) + current_frame - len(last_state_changes) + max(load_dist_max_idx-int(self.occlusion_window_size/2), 0) +1
        temp_df["orientation_buffer"] = temp_df["orientation_buffer"].replace({"front": False, "back":True})

        # add padding:
        pad = min(int(self.occlusion_window_size/2), len(temp_df["orientation_buffer"])) # Frames = 24/2
        pad_kernel = np.ones(pad).astype(bool)
        orientation_buffer_temp = list(np.convolve(list(temp_df["orientation_buffer"]), pad_kernel, 'same'))

        # reintroduce padding to temp_df
        temp_df["orientation_buffer"] = orientation_buffer_temp
        temp_df["orientation_buffer_going_true"] = temp_df['orientation_buffer'].shift(1, fill_value=False).apply(int).diff() == 1
        temp_df["orientation_buffer_going_false"] = temp_df['orientation_buffer'].apply(int).diff() == -1
        orientation_buffer_going_true = True in temp_df["orientation_buffer_going_true"].tolist() or all(temp_df["orientation_buffer_going_true"].tolist()) is True
        orientation_buffer_going_false = True in temp_df["orientation_buffer_going_false"].tolist() # or all(temp_df["orientation_buffer_going_false"].tolist()) is False

        # # Add padded to DataFrame
        # for i in temp_df.index:
        #     # print("edit rtg_padded")
        #     if temp_df["orientation_buffer"][i] == True:
        #         try:
        #             self.CACHE[i]["rtg_padded"] = True
        #         except KeyError:
        #             # print("edit rtg_padded exception")
        #             self.CACHE[i] = {"rtg_padded": True}

        # Drop: delete drop
        if contact_state_change == "drop" and orientation_buffer_going_true and not orientation_buffer_going_false:
            contact_state_change = False
            self.last_state_changes[last_state_change_idx] = False
            self.occlusion_detected = True
            print("Body occlusion at "+str(contact_frame_idx)+", contact state change (drop) deleted.")
            # autostopper
            self.last_going_false_index = temp_df.where(temp_df["orientation_buffer_going_false"]).last_valid_index()
            self.frames_since_last_occlusion = 0

        # Lift: delete the next lift
        if contact_state_change == "lift" and self.occlusion_detected:
            contact_state_change = False
            self.last_state_changes[last_state_change_idx] = False
            self.occlusion_detected = False
            print("Body occlusion finished at "+str(contact_frame_idx)+", contact state change (lift) deleted.")
            self.frames_since_last_occlusion = 0

        return contact_state_change
