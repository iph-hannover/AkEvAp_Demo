import sys
sys.path.insert(0, 'tram')

import os
import numpy as np
import torch

from lib.core import constants
from lib.core.config import update_cfg
from lib.models.hmr_vimo import HMR_VIMO

from PIL import Image
from collections import deque
from torch.utils.data import default_collate
from torchvision.transforms import Normalize, ToTensor, Compose
from torchvision.transforms.functional import resized_crop

class PoseEstimator:

    def __init__(self):
        self.logging = False
        self.log_dir = 'tmp/poses/'
        self.count = 0
        self.results = {'pred_cam': [], 'pred_pose': [], 'pred_shape': [], 'pred_rotmat': [], 'pred_trans': [], 'frame': []}

        # VIMO
        self.cfg = update_cfg('./tram/configs/config_vimo.yaml')
        self.cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = HMR_VIMO(self.cfg)
        ckpt = torch.load('./tram/data/pretrain/vimo_checkpoint.pth.tar', map_location='cpu', weights_only=True)
        mess = self.model.load_state_dict(ckpt['model'], strict=False)
        self.model = self.model.to('cuda')
        _ = self.model.eval()

        self.last_items = deque(maxlen=16)
        self.last_box = None
        self.normalize_img = Compose([ToTensor(), Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)])
        self.crop_size = 256

        # Indexing with the following mapping results in a compacted joint list in the order of the target skeleton format,
        # where joints without any correspondence are missing.
        # Therefore, we need to select the joints of interest before assigning (subset of joints for which we have correspondences)
        self.smpl2apple = [15, 12, 17, 19, 21, 16, 18, 20,  0,  2,  5,  8,  1,  4,  7]  # smpl.map_to(openpose)  # iOS_2D is a subset of OpenPose, so we can use that mapping
        self.joints_with_correspondence = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]  # openpose.joints_with_correspondences(smpl)

    def predict(self, image, compute_only=False, cache_input_only=False):
        do_cache_input = not compute_only
        do_compute = not cache_input_only
        height, width = image.shape[0], image.shape[1]

        if do_cache_input:
            img_focal = 1400
            img_center = [width/2, height/2]

            box_init = np.asarray([0, 0, width, height, 1.0])
            box = box_init if self.count < 16 else self.last_box

            xmin, ymin, xmax, ymax = box[:4]
            w, h = xmax-xmin, ymax-ymin
            center = np.asarray([xmin+w/2, ymin+h/2], dtype=np.float32)
            scale = np.asarray(max(w, h) / 200 * 1.2, dtype=np.float32)

            left, top = center - 200*scale/2  # 200px is reference patch size (used internally in  the model, so we cannot remove it here...)
            img_crop = resized_crop(img=Image.fromarray(image), top=top, left=left, height=200*scale, width=200*scale, size=[self.crop_size, self.crop_size])
            img_crop = self.normalize_img(img_crop)
            item = {
                'img': img_crop.float(),
                'img_idx': self.count,
                'scale': scale.astype(np.float32),
                'center': center.astype(np.float32),
                'img_focal': np.asarray(img_focal, dtype=np.float32),
                'img_center': np.asarray(img_center, dtype=np.float32),
            }
            self.last_items.append(item)

        if len(self.last_items) < 16 or not do_compute:  # we need a batch of 16 images
            self.count += 1
            return None, None

        batch = default_collate(self.last_items)

        with torch.no_grad():
            batch = {k: v.to(self.cfg.DEVICE) for k, v in batch.items() if type(v) == torch.Tensor}
            out, _ = self.model.forward(batch)

        # we are only interested in the "current" pose of the sliding window
        out = {k: v[-1] for k, v in out.items()}

        if self.logging:
            # store results
            self.results['pred_cam'].append(out['pred_cam'].cpu().numpy())
            self.results['pred_pose'].append(out['pred_pose'].cpu().numpy())
            self.results['pred_shape'].append(out['pred_shape'].cpu().numpy())
            self.results['pred_rotmat'].append(out['pred_rotmat'].cpu().numpy())
            self.results['pred_trans'].append(out['trans_full'].cpu().numpy())
        self.count += 1

        pred_rotmat = out['pred_rotmat'].unsqueeze(dim=0)  # B=1 x 24 x 3 x 3
        pred_shape = out['pred_shape'].unsqueeze(dim=0)  # B=1 x 10
        pred_trans = out['trans_full']  # B=1 x 3

        # TODO: SMPL was already queried within HMR_VIMO -> reuse results rather than recomputing
        pred = self.model.smpl(body_pose=pred_rotmat[:, 1:],
                               global_orient=pred_rotmat[:, [0]],
                               betas=pred_shape,
                               transl=pred_trans,
                               pose2rot=False,
                               default_smpl=True)

        #pred_vert = pred.vertices
        pred_j3d = pred.joints[:, :24]
        pred_camt = torch.zeros(3).unsqueeze(dim=0).to(self.cfg.DEVICE)  # B=1 x 3
        pred_camr = torch.eye(3).unsqueeze(dim=0).to(self.cfg.DEVICE)  # B=1 x 3 x 3
        if pred_j3d[0, 0, 2].item() > 5:  # if depth > 5m we assume a failed detection
            self.reset()
            return None, None

        #pred_vert_w = torch.einsum('bij,bnj->bni', pred_camr, pred_vert) + pred_camt[:, None]
        pred_j3d_w = torch.einsum('bij,bnj->bni', pred_camr, pred_j3d) + pred_camt[:, None]

        # project to 2D
        intrinsics = self.build_intrinsics(dict(fx=batch['img_focal'][-1],
                                                fy=batch['img_focal'][-1],
                                                cx=batch['img_center'][-1][0],
                                                cy=batch['img_center'][-1][1]))
        pred_j2d_proj = (intrinsics @ pred_j3d_w[0].T).T  # 24 x 3
        pred_j2d_proj = pred_j2d_proj[:, :2] / pred_j2d_proj[:, [2]]
        pred_j2d_proj = pred_j2d_proj[:, :2]
        pred_j2d_proj = pred_j2d_proj.cpu().numpy()

        pts = np.zeros([19, 2], dtype=np.float32)  # empty pose for apple's 2D skeleton
        pts[self.joints_with_correspondence] = pred_j2d_proj[self.smpl2apple, :]
        pts_3d = np.zeros([19, 3], dtype=np.float32)  # empty pose for apple's 2D skeleton
        pts_3d[self.joints_with_correspondence] = pred_j3d_w[0][self.smpl2apple, :].cpu().numpy()

        self.last_box = np.asarray([pred_j2d_proj.min(axis=0)[0], pred_j2d_proj.min(axis=0)[1],
                                    pred_j2d_proj.max(axis=0)[0], pred_j2d_proj.max(axis=0)[1], 1.0], dtype=np.float32)
        self.last_box[2] += height*0.1  # add some more padding for head
        self.last_box[:2] = np.maximum(self.last_box[:2], [0, 0])    # clip lower bounds
        self.last_box[2:4] = np.minimum(self.last_box[2:4], [width, height])  # clip upper bounds
        #print(np.array2string(self.last_box, precision=0, separator=',', suppress_small=True, formatter={'float_kind': lambda x: f'{f"{x:.0f}":>5}'}))

        return pts.astype(int), pts_3d

    def dump_data(self):
        # stack results
        self.results['pred_cam'] = np.stack(self.results['pred_cam'], axis=0)
        self.results['pred_pose'] = np.stack(self.results['pred_pose'], axis=0)
        self.results['pred_shape'] = np.stack(self.results['pred_shape'], axis=0)
        self.results['pred_rotmat'] = np.stack(self.results['pred_rotmat'], axis=0)
        self.results['pred_trans'] = np.stack(self.results['pred_trans'], axis=0)

        file_name = os.path.join(self.log_dir, f'results.npz')
        np.savez(file_name, **self.results)

        with open(file_name, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            pred_cam = data['pred_cam']
            pred_pose = data['pred_pose']
            pred_shape = data['pred_shape']
            pred_rotmat = data['pred_rotmat']
            pred_trans = data['pred_trans']

    def build_intrinsics(self, intr):
        mat = torch.zeros([3, 3], device=self.cfg.DEVICE)
        mat[0, 0] = intr['fx']
        mat[1, 1] = intr['fy']
        mat[0, 2] = intr['cx']
        mat[1, 2] = intr['cy']
        mat[2, 2] = 1
        return mat

    def reset(self):
        self.count = 0
        self.last_box = None
        self.last_items.clear()
