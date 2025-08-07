import numpy as np
import re
import sys
from skeletons import JOINTS


def parse_ios3D_data_new(data, trial, load_2D_projections=False):  # for new json conform dumps
    points3d = []
    heights = []
    n_joints = len(JOINTS['3D'])
    dimensionality = 2 if load_2D_projections else 3
    for det_idx, det in enumerate(data):
        if len(det['poses']) < 1:
            print(f"WARNING: {trial}: Before {det['frame'] if 'frame' in det.keys() else -1} {1} detection is missing")
            points3d.append(np.zeros([n_joints, dimensionality]))
        elif len(det['poses']) == 1:
            pts = []
            for joint in JOINTS['3D']:
                if load_2D_projections:
                    pt = det['poses'][0][f'joints2D'][f"human_{joint}_3D"]
                    pts.append([1-pt['x'], pt['y']])  # need to rotate image 180°
                else:
                    pt = det['poses'][0][f'joints'][f"human_{joint}_3D"]
                    pts.append([pt['x'], pt['y'], pt['z']])
            pts = np.asarray(pts)  # n_joints x 3

            points3d.append(pts)
            if not load_2D_projections:
                heights.append(det['poses'][0][f'bodyHeight'])
        else:
            print("Should never happen!")  # 3D detection always returns max one skeleton
    points3d = np.stack(points3d, axis=0).reshape([len(points3d), -1, 2 if load_2D_projections else dimensionality])

    return points3d


def parse_alphapose3D_data(data, trial):
    points2d = []
    dif = 0
    for det_idx, det in enumerate(data):
        match = re.match("[a-z]*_*([0-9]+).[a-z]+", det['image_id'])
        img_idx = int(match.groups()[0])
        if det_idx - img_idx != dif:
            det_dif = det_idx - img_idx - dif
            if det_dif < 0:
                print(f"WARNING: {trial}: Before {det['image_id']} {-det_dif} detections are missing")
                # add empty detections
                for _ in range(-det_dif):
                    points2d.append(np.zeros([24, 3]))
                # add current detection
                if len(det['pred_xyz_jts']) != 24:
                    print(img_idx, len(det['pred_xyz_jts']))
                points2d.append(np.asarray(det['pred_xyz_jts']).reshape(-1, 3))
            elif det_dif > 0:
                assert det_dif == 1  # multiple images can have no detections, but multiple detections per frame can only occur in steps of size one (every entry contains max 1 detection)
                print(f"WARNING: {trial}: {det['image_id']} contains {det_dif} double detection")
                current_det = np.asarray(det['pred_xyz_jts']).reshape(-1, 3)
                previous_det = points2d[-1]
                reference_det = points2d[-2] if len(points2d) >= 2 else previous_det  # if idx=0 got multiple detections => no previous detection exists...
                dist_c = np.linalg.norm(current_det - reference_det, axis=1).sum()
                dist_p = np.linalg.norm(previous_det - reference_det, axis=1).sum()
                if dist_c < dist_p:
                    points2d[-1] = current_det
                    print(f"Dropping the first of both detections due to higher distance to the pose in the last frame: {dist_c:.2f} vs. {dist_p:.2f}")
                else:
                    print(f"Dropping the second of both detections due to higher distance to the pose in the last frame: {dist_c:.2f} vs. {dist_p:.2f}")
            else:
                print("Should never happen!")
            dif = det_idx - img_idx
        else:
            if len(det['pred_xyz_jts']) != 24:
                print(img_idx, len(det['pred_xyz_jts']))
            points2d.append(np.asarray(det['pred_xyz_jts']).reshape(-1, 3))
    points2d = np.stack(points2d, axis=0).reshape([len(points2d), -1, 3])

    # if there is no skeleton detected in the last frames, the array is shorter than #frames
    return points2d
