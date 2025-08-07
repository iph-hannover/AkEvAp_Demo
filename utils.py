import numpy as np
from scipy.spatial.transform import Rotation
from types import SimpleNamespace
from skeletons import JOINTS
import time

class GetTime:
    def __init__(self, name="", verbose=False):
        self.name = name
        self.verbose = verbose
    def __enter__(self):
        if self.verbose:
            import torch
            torch.cuda.synchronize()
            self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose:
            import torch
            torch.cuda.synchronize()
            self.end = time.time()
            self.duration = self.end - self.start
            print(f"{self.name}: {self.duration:.4f} seconds")


def get_joint_indices(skeleton_type):
    JOINTS_ = SimpleNamespace()
    JOINTS_.ROOT = JOINTS[skeleton_type].index('root')
    if skeleton_type == '3D':
        JOINTS_.R_WRIST = JOINTS[skeleton_type].index('right_wrist')
        JOINTS_.L_WRIST = JOINTS[skeleton_type].index('left_wrist')
        JOINTS_.R_ELBOW = JOINTS[skeleton_type].index('right_elbow')
        JOINTS_.L_ELBOW = JOINTS[skeleton_type].index('left_elbow')
        JOINTS_.R_FOOT = JOINTS[skeleton_type].index('right_ankle')
        JOINTS_.L_FOOT = JOINTS[skeleton_type].index('left_ankle')
        JOINTS_.R_HIP = JOINTS[skeleton_type].index('right_hip')
        JOINTS_.L_HIP = JOINTS[skeleton_type].index('left_hip')
        JOINTS_.NECK = JOINTS[skeleton_type].index('center_shoulder')
    elif skeleton_type == '2D':
        JOINTS_.R_WRIST = JOINTS[skeleton_type].index('right_hand_joint')
        JOINTS_.L_WRIST = JOINTS[skeleton_type].index('left_hand_joint')
        JOINTS_.R_ELBOW = JOINTS[skeleton_type].index('right_forearm_joint')
        JOINTS_.L_ELBOW = JOINTS[skeleton_type].index('left_forearm_joint')
        JOINTS_.R_FOOT = JOINTS[skeleton_type].index('right_foot_joint')
        JOINTS_.L_FOOT = JOINTS[skeleton_type].index('left_foot_joint')
        JOINTS_.R_HIP = JOINTS[skeleton_type].index('right_upLeg_joint')
        JOINTS_.L_HIP = JOINTS[skeleton_type].index('left_upLeg_joint')
        JOINTS_.NECK = JOINTS[skeleton_type].index('neck_1_joint')
    return JOINTS_


def compute_load_center(points, skeleton_type):
    JOINTS_ = get_joint_indices(skeleton_type)

    # load center is defined as mid-point of large middle knuckles (e.g. hand joint in SMPL model)

    arm_r = points[JOINTS_.R_WRIST] - points[JOINTS_.R_ELBOW]
    arm_l = points[JOINTS_.L_WRIST] - points[JOINTS_.L_ELBOW]
    mid_knuckle_r = points[JOINTS_.R_WRIST] + arm_r * 1/3  # => add 1/3 of arm length as offset to each wrist joint
    mid_knuckle_l = points[JOINTS_.L_WRIST] + arm_l * 1/3
    # mid_knuckle_r = points[JOINTS_.R_WRIST] + arm_r / np.linalg.norm(arm_r) * 0.07  # => add 7cm of arm length as offset to each wrist joint
    # mid_knuckle_l = points[JOINTS_.L_WRIST] + arm_l / np.linalg.norm(arm_l) * 0.07
    load_center = (mid_knuckle_r + mid_knuckle_l) / 2
    #print("load center:", load_center, (points[JOINTS_.R_WRIST] + points[JOINTS_.L_WRIST]) / 2)
    return load_center

def normalize_pose(points3d, skeleton_type):
    JOINTS_ = get_joint_indices(skeleton_type)

    # points3d are provided in camera coordinates -> camera is fixed at the origin (but y axis is pointing down and x to the right)
    z180 = Rotation.from_euler('z', 180, degrees=True).as_matrix()
    points3d = (z180 @ points3d.T).T  # => rotate the camera coordinate system to align with world coordinate frame first

    # move the whole scene so that the origin is located at the root joint for subsequent rotation
    trans_root = points3d[[JOINTS_.ROOT]]
    points3d = points3d - trans_root

    # rotate the whole scene around y axis so that the hips are in x direction (y & z lay in the sagittal plane, x is in the frontal plane (but it is not the normal vector of the sagittal plane as it can show slightly upwards))
    hip_bone = points3d[JOINTS_.L_HIP] - points3d[JOINTS_.R_HIP]
    ang = np.arctan2(hip_bone[2], hip_bone[0]) * 180 / np.pi  # arctan2(y, x) computes angle regarding (x=1, y=0) => hip_bone should align with x-axis
    R = Rotation.from_euler('y', ang, degrees=True)
    points3d = (R.as_matrix() @ points3d.T).T

    y_offset = min(points3d[JOINTS_.R_FOOT][1], points3d[JOINTS_.L_FOOT][1])
    points3d[:, 1] -= y_offset  # always assume one foot on the floor

    # move the whole scene so that the origin is located at the mid-point between the inner ankle bones => e.g. mid-point of Ankle/Foot joints in SMPL model
    root_offset = np.asarray([0, points3d.min(axis=0)[1], 0], dtype=np.float32)  # => origin is located on the floor plane below the hips (at the intersection of sagittal, frontal plane and floor)
    ankle_mid = (points3d[JOINTS_.R_FOOT] + points3d[JOINTS_.L_FOOT]) / 2
    # print("root offset:", root_offset, "ankle_mid:", ankle_mid, "dist:", np.linalg.norm(root_offset - ankle_mid))
    feet_direction = np.cross(points3d[JOINTS_.R_FOOT] - points3d[JOINTS_.L_FOOT], np.asarray([0, 1., 0]))
    feet_direction /= np.linalg.norm(feet_direction)
    feet_direction = feet_direction if feet_direction[2] > 0 else -feet_direction  # invert direction if necessary so that it always points towards the toes (z-direction must always be positive)
    ankle_mid = ankle_mid + feet_direction * 0.07  # small correction for mid-ankle towards mid-point between the inner ankle bones
    #print("ankle mid:", ankle_mid - feet_direction * 0.07, ankle_mid)
    points3d = points3d - ankle_mid

    return points3d

def extract_patch(image, center, patch_size=(200, 200)):
    height, width = image.shape[:2]

    # calculate patch boundaries
    x_min = int(center[0] - patch_size[0] // 2)
    x_max = int(center[0] + patch_size[0] // 2)
    y_min = int(center[1] - patch_size[1] // 2)
    y_max = int(center[1] + patch_size[1] // 2)

    # add padding if patch exceeds boundary
    left_pad = int(max(0 - x_min, 0))
    right_pad = int(max(x_max - width, 0))
    top_pad = int(max(0 - y_min, 0))
    bottom_pad = int(max(y_max - height, 0))

    if len(image.shape) == 3:
        pad_width = [[top_pad, bottom_pad], [left_pad, right_pad], [0, 0]]
        padded_image = np.pad(image, pad_width=pad_width, mode='constant')
        patch = padded_image[y_min+top_pad:y_max+top_pad, x_min+left_pad:x_max+left_pad, :]
    elif len(image.shape) == 2:
        pad_width = [[top_pad, bottom_pad], [left_pad, right_pad]]
        padded_image = np.pad(image, pad_width=pad_width, mode='constant')
        patch = padded_image[y_min+top_pad:y_max+top_pad, x_min+left_pad:x_max+left_pad]
    # assert (patch.shape == np.asarray([patch_size[1], patch_size[0], 3])).all()

    return patch
