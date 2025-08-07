import os
import numpy as np
import torch

from collections import deque
from enum import Enum
import pandas as pd
from utils import get_joint_indices
from scipy.spatial.transform import Rotation

from LMM_classification.SimpleClassificator import SimpleClassificator
from LMM_classification.ResidualPoseClassificator import ResidualPoseClassificator
from types import SimpleNamespace

class PoseTypeLMM(Enum):
    NORMAL = 0
    SLIGHT_OFFSET = 1
    STRONG_OFFSET = 2
    KNEEING = 3
    NONE = 4

def smooth_categorical(df):
    def fn(x):
        return x.mode()[0]  # The mode is the value that appears most often. There can be multiple modes.
    return pd.Series(df, dtype='category').cat.codes.rolling(7, center=True, min_periods=0).apply(fn).map(dict(enumerate(df.cat.categories)))

class LMMClassificator:

    def __init__(self):
        self.logging = True
        self.log_dir = 'tmp/lmm/'
        self.count = 0
        self.results = {'pred_label': [], 'abs pose diff': []}

        cfg = SimpleNamespace(emb_size=256, use3D=True, num_joints=14, num_classes=4, dropout_rate=0.3, weight_init=True, num_residual_blocks=4, activation_fn='leaky_relu')
        # self.model = SimpleClassificator(cfg)
        # weights = torch.load(f'./LMM_classification/models/poseType_weights.pth', weights_only=True)
        self.model = ResidualPoseClassificator(cfg)
        weights = torch.load(f'./LMM_classification/models/poseType_weights_residual.pth', weights_only=True)
        mess = self.model.load_state_dict(weights, strict=False)
        self.model = self.model.to('cuda')
        _ = self.model.eval()

        self.last_results = deque(maxlen=20)

        # Indexing with the following mapping results in a compacted joint list in the order of the target skeleton format,
        # where joints without any correspondence are missing.
        # Therefore, we need to select the joints of interest before assigning (subset of joints for which we have correspondences)
        self.apple_2D_to_3D = [ 8,  9, 10, 11, 12, 13, 14,  1,  0,  5,  6,  7,  2,  3,  4]  # iOS_2D.map_to(iOS_3D)
        self.joints_with_correspondence = [ 0,  1,  2,  3,  4,  5,  6,  8,  9, 11, 12, 13, 14, 15, 16]  # iOS_3D.joints_with_correspondences(iOS_2D)

    def predict(self, pts_3d):

        # NOTE: currently not required as we already pass iOS_3D points
        # # normalized points are provided: move the whole scene so that the origin is located at the root joint
        # JOINTS_ = get_joint_indices('2D')
        # trans_root = pts_3d[[JOINTS_.ROOT]]
        # pts_3d_rr = pts_3d - trans_root
        # 
        # points3d = np.zeros([17, 3], dtype=np.float32)  # empty pose for apple's 3D skeleton
        # points3d[self.joints_with_correspondence] = pts_3d_rr[self.apple_2D_to_3D, :] * 1000
        points3d = (pts_3d * 1000).astype(np.float32)

        # model does not use head/nose/spine joints
        points3d = points3d[[jid not in [7, 9, 10] for jid in range(len(points3d))]]

        inputs = torch.from_numpy(points3d).flatten().unsqueeze(dim=0).unsqueeze(dim=1)  # B=1 x win_size=1 * num_joints=17 * 3
        inputs = inputs.reshape(inputs.shape[0], -1)  # B=1 x win_size=1 * num_joints=17 * 3
        out = self.model(inputs.to('cuda'))
        _, predicted = torch.max(out['logits'], 1)
        predicted = predicted.item()

        if self.logging:
            # store results
            self.results['pred_label'].append(predicted)
        
        self.count += 1
        self.last_results.append(predicted)

        pred_convolved = np.asarray(smooth_categorical(pd.Series(self.last_results, dtype='category')))

        type_raw = PoseTypeLMM(predicted)
        type_smoothed = PoseTypeLMM(pred_convolved[-1])

        return type_smoothed, type_raw

    def dump_data(self):
        # stack results
        self.results['pred_label'] = np.stack(self.results['pred_label'], axis=0)

        file_name = os.path.join(self.log_dir, f'results.npz')
        np.savez(file_name, **self.results)

        with open(file_name, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            pred_label = data['pred_label']

    def reset(self):
        self.last_results.clear()
