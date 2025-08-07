import sys
sys.path.insert(0, 'hand_object_detector')

import os
import numpy as np
import torch

from collections import deque

from model.utils.config import cfg, cfg_from_file
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import vis_detections_filtered_objects_PIL
from model.faster_rcnn.resnet import resnet
from demo import _get_image_blob
import utils


class ContactDetector:

    def __init__(self, skeleton_type, initial_frame_idx):
        self.logging = False
        self.log_dir = 'tmp/hand_contact/'
        self.count = initial_frame_idx
        self.results = {'frame': [], 'last_scores': [], 'boxes_obj': [], 'scores_obj': [], 'boxes': [], 'scores': [], 'contact_states': [], 'offset_vectors': [], 'lr': []}

        # global params
        self.thresh_hand = 0.5      # threshold for hand detections
        self.thresh_object = 0.65   # threshold for object-contact
        self.pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])
        self.class_agnostic = False

        # initialize model
        self.model_path = "./hand_object_detector/models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth"
        self.fasterRCNN = resnet(self.pascal_classes, 101, pretrained=False, class_agnostic=self.class_agnostic)
        self.fasterRCNN.create_architecture()
        checkpoint = torch.load(self.model_path, weights_only=True)
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        cfg_from_file("./hand_object_detector/cfgs/res101.yml")
        self.fasterRCNN.cuda()
        self.fasterRCNN.eval()

        # initialize the tensor holder
        self.im_data = torch.FloatTensor(1)
        self.im_info = torch.FloatTensor(1)
        self.num_boxes = torch.LongTensor(1)
        self.gt_boxes = torch.FloatTensor(1)
        self.box_info = torch.FloatTensor(1)

        # ship to cuda
        if torch.cuda.is_available():
            self.im_data = self.im_data.cuda()
            self.im_info = self.im_info.cuda()
            self.num_boxes = self.num_boxes.cuda()
            self.gt_boxes = self.gt_boxes.cuda()

        self.skeleton_type = skeleton_type
        self.contact_state = 'none'
        self.last_scores = deque(maxlen=20)
        self.last_load_centers = deque(maxlen=30)

    def predict(self, image, pts_3d, pts_2d):
        if pts_3d is None:
            score = 0
            self.last_scores.append(score)
            return False

        load_center = utils.compute_load_center(pts_3d, self.skeleton_type)
        load_center_proj = np.asarray([load_center[0], 0, load_center[2]])
        self.last_load_centers.append(load_center_proj)
        load_center_2d = utils.compute_load_center(pts_2d, self.skeleton_type)

        # prepare input patch
        patch_size = [300, 300]
        patch = utils.extract_patch(image, patch_size=patch_size, center=load_center_2d)
        im_blob, im_scales = _get_image_blob(patch)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        self.im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        self.im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        self.gt_boxes.resize_(1, 1, 5).zero_()
        self.num_boxes.resize_(1).zero_()
        self.box_info.resize_(1, 1, 5).zero_()

        # query model
        with torch.no_grad():
            outputs = self.fasterRCNN(self.im_data, self.im_info, self.gt_boxes, self.num_boxes, self.box_info)

        hand_dets, obj_dets = self.postprocess(outputs, im_scales)
        if hand_dets is None:
            hand_dets = np.asarray([[-1, -1, -1, -1, 0, 0, -1, -1, -1, 0]])
        if obj_dets is None:
            obj_dets = np.asarray([[-1, -1, -1, -1, 0, 0, -1, -1, -1, 0]])
        # in case of multiple detections, always use the most confident one for class=3 (contact with portable object)
        hand_det = hand_dets[hand_dets[:, 5] == 3]
        if len(hand_det):
            box = hand_det[0, :4]
            score = hand_det[0, 4]
            contact_state_idx = hand_det[0, 5]  # hand contact state info => 0: 'No Contact', 1: 'Self Contact', 2: 'Another Person', 3: 'Portable Object', 4: 'Stationary Object'
            offset_vector = hand_det[0, 6:9]    # offset vector (factored into a unit vector and a magnitude)
            lr = hand_det[0, 9]                 # hand side info => 0: 'Left', 1: 'Right'
        else:
            score = 0
            contact_state_idx = 0
        # in case of multiple detections, always use the most confident one (all entries except from bbox and score cannot be interpreted by design as they were not trained for objects)
        obj_det = obj_dets[[np.argmax(obj_dets[:, 4])]]
        obj_box = obj_det[0, :4]
        obj_score = obj_det[0, 4]

        self.last_scores.append(score if contact_state_idx == 3 and obj_score > self.thresh_object else 0)  # class 3 == contact with portable object

        if self.logging:
            full_image = False
            if full_image:
                x_min = int(load_center_2d[0] - patch_size[0] // 2)
                y_min = int(load_center_2d[1] - patch_size[1] // 2)
                obj_dets_ = np.concatenate([obj_dets[:, :4] + [x_min, y_min, x_min, y_min], obj_dets[:, 4:]], axis=1)  # bbox format: [x1, y1, x2, y2]
                hand_dets_ = np.concatenate([hand_dets[:, :4] + [x_min, y_min, x_min, y_min], hand_dets[:, 4:]], axis=1)
                obj_dets_ = obj_dets_[obj_dets_[:, 4] > 0]        # filter zero confidence entries
                # hand_dets_ = hand_dets_[hand_dets_[:, 5] == 3]  # draw hand only when portable object contact
                obj_dets_ = obj_dets_ if len(obj_dets_) else None
                hand_dets_ = hand_dets_ if len(hand_dets_) else None
                out_im = vis_detections_filtered_objects_PIL(image[:, :, ::-1], obj_dets_, hand_dets_, self.thresh_hand, self.thresh_object, font_path='./hand_object_detector/lib/model/utils/times_b.ttf')
            else:
                out_im = vis_detections_filtered_objects_PIL(patch[:, :, ::-1], obj_dets, hand_dets, self.thresh_hand, self.thresh_object, font_path='./hand_object_detector/lib/model/utils/times_b.ttf')
            out_im.save(f'{self.log_dir}/{self.count:08d}.png')

            # store results
            self.results['frame'].append(self.count)
            self.results['last_scores'].append(self.last_scores[-1])
            self.results['boxes_obj'].append(np.asarray(obj_dets[:, :4]))
            self.results['scores_obj'].append(np.asarray(obj_dets[:, 4]))
            self.results['boxes'].append(np.asarray(hand_dets[:, :4]))
            self.results['scores'].append(np.asarray(hand_dets[:, 4]))
            self.results['contact_states'].append(np.asarray(hand_dets[:, 5]))    # hand contact state info => 0: 'No Contact', 1: 'Self Contact', 2: 'Another Person', 3: 'Portable Object', 4: 'Stationary Object'
            self.results['offset_vectors'].append(np.asarray(hand_dets[:, 6:9]))  # offset vector (factored into a unit vector and a magnitude)
            self.results['lr'].append(np.asarray(hand_dets[:, 9]))                # hand side info => 0: 'Left', 1: 'Right'
        self.count += 1

        last = np.asarray(self.last_scores)
        avg = last.mean()
        new_state = 'carry' if len(last) > 3 and avg > self.thresh_hand*0.75 else 'none'
        if new_state == 'carry' and self.contact_state == 'none':
            change = 'lift'
        elif new_state == 'none' and self.contact_state == 'carry':
            change = 'drop'
        else:
            change = False
        self.contact_state = new_state

        return change

    def dump_data(self):
        # cannot stack as shapes might differ (0/1/2 detections per frame) => convert to object array as container
        self.results['boxes_obj'] = np.array(self.results['boxes_obj'], dtype=object)
        self.results['scores_obj'] = np.array(self.results['scores_obj'], dtype=object)
        self.results['boxes'] = np.array(self.results['boxes'], dtype=object)
        self.results['scores'] = np.array(self.results['scores'], dtype=object)
        self.results['contact_states'] = np.array(self.results['contact_states'], dtype=object)
        self.results['offset_vectors'] = np.array(self.results['offset_vectors'], dtype=object)
        self.results['lr'] = np.array(self.results['lr'], dtype=object)
        file_name = os.path.join(self.log_dir, f'results.npz')
        np.savez(file_name, **self.results)

        with open(file_name, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            frame_idx = list(data['frame'])
            last_scores = list(data['last_scores'])
            boxes_obj = list(data['boxes_obj'])
            scores_obj = list(data['scores_obj'])
            boxes = list(data['boxes'])
            scores = list(data['scores'])
            contact_states = list(data['contact_states'])
            offset_vectors = list(data['offset_vectors'])
            lr = list(data['lr'])


    if torch.cuda.is_available() > 0:
        bbox_normalize_stds = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
        bbox_normalize_means = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
    else:
        bbox_normalize_stds = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        bbox_normalize_means = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

    def postprocess(self, outputs, im_scales):
        rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, RCNN_loss_cls, RCNN_loss_bbox, rois_label, loss_list = outputs

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        # extract predicted params
        contact_vector = loss_list[0][0]          # hand contact state info
        offset_vector = loss_list[1][0].detach()  # offset vector (factored into a unit vector and a magnitude)
        lr_vector = loss_list[2][0].detach()      # hand side info (left/right)

        # get hand contact
        _, contact_indices = torch.max(contact_vector, 2)
        contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

        # get hand side
        lr = torch.sigmoid(lr_vector) > 0.5
        lr = lr.squeeze(0).float()

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                box_deltas = bbox_pred.data
                box_deltas = box_deltas.view(-1, 4) * ContactDetector.bbox_normalize_stds + ContactDetector.bbox_normalize_means
                if self.class_agnostic:
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(1, -1, 4 * len(self.pascal_classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, self.im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        obj_dets, hand_dets = None, None

        for j in range(1, len(self.pascal_classes)):
            # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
            if self.pascal_classes[j] == 'hand':
                inds = torch.nonzero(scores[:, j] > self.thresh_hand).view(-1)
            elif self.pascal_classes[j] == 'targetobject':
                inds = torch.nonzero(scores[:, j] > self.thresh_object).view(-1)

            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:,j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if self.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if self.pascal_classes[j] == 'targetobject':
                    obj_dets = cls_dets.cpu().numpy()
                if self.pascal_classes[j] == 'hand':
                    hand_dets = cls_dets.cpu().numpy()

        return hand_dets, obj_dets
