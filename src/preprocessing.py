import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch.nn as nn
import cv2
from config import data_dir, images_train_dir, images_val_dir, labels_train_dir, labels_val_dir
import os

class FaceDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_list, max_boxes= 10, transform= None):
        self.image_dir= image_dir
        self.label_dir= label_dir
        self.image_list= image_list
        self.transform= transform
        self.max_boxes= max_boxes

    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        
        image_name= self.image_list[idx]
        image_path= os.path.join(self.image_dir, image_name)
        label_path= os.path.join(self.label_dir, image_name.rsplit('.', 1)[0] + '.txt')

        image= cv2.imread(image_path)
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width= image.shape[:2]
        bounding_boxes= []


        if self.transform:
            tensor_image= torch.from_numpy(image).permute(2, 0, 1).float()
            tensor_image= self.transform(tensor_image)
            resized_height, resized_width= tensor_image.shape[1:3]
            with open(label_path, 'r') as f:
                for line in f:
                    _, x, y, box_width, box_height= map(float, line.strip().split())
                    x1= int((x - box_width / 2) * resized_width)
                    y1= int((y - box_height / 2) * resized_height)
                    x2= int((x + box_width / 2) * resized_width)
                    y2= int((y + box_height / 2) * resized_height)
                    bounding_boxes.append([x1, y1, x2, y2])
        num_curr_boxes= len(bounding_boxes)
        if num_curr_boxes < self.max_boxes:
            padding= np.zeros((self.max_boxes - num_curr_boxes, 4), dtype= np.float32)
            padded_boxes= np.concatenate((np.array(bounding_boxes, dtype= np.float32),
                                           padding), axis= 0)
            
        else:
            padded_boxes= np.array(bounding_boxes[:self.max_boxes], dtype=np.float32)
        
        boxes_tensor= torch.tensor(padded_boxes, dtype= torch.float32)

        return {
            'image': tensor_image,
            'boxes': boxes_tensor
        }

def generate_anchor_boxes(feature_map_shape, anchor_scales, anchor_ratios,
                     stride, num_anchors_per_location):
    anchor_base_wh= []
    for scale in anchor_scales:
        area= scale * scale
        for ratio in anchor_ratios:
            width= np.sqrt(area * ratio)
            height= np.sqrt(area / ratio)
            anchor_base_wh.append((width, height))
    
    fm_height, fm_width= feature_map_shape
    # Get centers in feature map space
    y_feature_map, x_feature_map= np.mgrid[0:fm_height, 0:fm_width]
    centers_x_fm= x_feature_map.ravel()
    centers_y_fm= y_feature_map.ravel()
    
    # Get centers in image space
    centers_x_img= (centers_x_fm + 0.5) * stride
    centers_y_img= (centers_y_fm + 0.5) * stride
    
    # Repeat for all anchors per location
    centers_x_img= np.repeat(centers_x_img, num_anchors_per_location)
    centers_y_img= np.repeat(centers_y_img, num_anchors_per_location)
    
    # Create width-height pairs for each of the anchor types
    width_height_per_anchor= np.tile(anchor_base_wh, (fm_height * fm_width, 1))

    # Build anchor boxes
    anchor_width, anchor_height= width_height_per_anchor[:, 0], width_height_per_anchor[:, 1]
    x1= centers_x_img - anchor_width / 2.0
    y1= centers_y_img - anchor_height / 2.0
    x2= centers_x_img + anchor_width / 2.0
    y2= centers_y_img + anchor_height / 2.0

    x1= np.clip(x1, 0, 223)
    y1= np.clip(y1, 0, 223)
    x2= np.clip(x2, 0, 223)
    y2= np.clip(y2, 0, 223)
    anchors= np.stack([x1, y1, x2, y2], axis= 1)
    anchors_tensor= torch.tensor(anchors, dtype= torch.float32)
    return anchors_tensor

def filter_valid_bboxes(gt_boxes):
    valid_mask= (gt_boxes.sum(dim=1) > 0) & ((gt_boxes[:, 2] - gt_boxes[:, 0]) > 0) & ((gt_boxes[:, 3] - gt_boxes[:, 1]) > 0)
    return gt_boxes[valid_mask]

def calculate_iou(anchor_boxes, gt_boxes, eps: float= 1e-6, device: torch.device= None):
    # Get areas
    anchor_areas= (anchor_boxes[:, 2] - anchor_boxes[:, 0]) * (anchor_boxes[:, 3] - anchor_boxes[:, 1])
    gt_areas= (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    # Get intersection info
    top_left_intersection= torch.maximum(anchor_boxes[:, None, :2], gt_boxes[None, :, :2])
    right_bottom_intersection= torch.minimum(anchor_boxes[:, None, 2:], gt_boxes[None, :, 2:])
    width_height_intersection= torch.clamp(right_bottom_intersection - top_left_intersection, min= 0)
    inter_area= width_height_intersection[:, :, 0] * width_height_intersection[:, :, 1]
    # Union
    union= anchor_areas[:, None] + gt_areas[None, :] - inter_area
    # IOU
    iou= inter_area / (union + eps)
    return iou


def match_anchors_to_gt(
        anchors,
        gt_boxes,
        iou_high_threshold: float = 0.6,
        iou_low_threshold: float = 0.3,
        num_samples: int = 64,
        pos_ratio: float = 0.5,
        anchors_per_gt: int = 16,
        device: torch.device = None
):
    if device is None:
        device = anchors.device

    num_anchors = anchors.shape[0]
    num_gt = gt_boxes.shape[0]

    iou = calculate_iou(anchors, gt_boxes)

    labels = torch.full((num_anchors,), -1, dtype=torch.long, device=device)

    # Foreground: for each GT pick its best anchors (up to anchors_per_gt)
    for gt_id in range(num_gt):
        gt_iou = iou[:, gt_id]
        cand = torch.where(gt_iou >= iou_high_threshold)[0]
        if cand.numel() == 0:                       # no anchor meets threshold
            cand = torch.tensor([gt_iou.argmax()], device=device)    # fall back to best
        if cand.numel() > anchors_per_gt:
            cand = cand[torch.randperm(cand.numel())[:anchors_per_gt]]
        labels[cand] = 1

    # Background
    bg_mask = (iou < iou_low_threshold).all(dim=1)
    labels[bg_mask] = 0

    # Global balanced sampling (unchanged)
    pos_target = int(num_samples * pos_ratio)
    neg_target = num_samples - pos_target

    pos_inds = torch.where(labels == 1)[0]
    if len(pos_inds) > 0:
        for idx in pos_inds:
            max_iou_for_this_anchor = iou[idx].max()
            
            # Check if this anchor meets threshold OR is a fallback assignment
            meets_threshold = max_iou_for_this_anchor >= iou_high_threshold
            
            # Check if this is a fallback case (best anchor for some GT)
            is_fallback = False
            for gt_id in range(num_gt):
                gt_iou = iou[:, gt_id]
                if gt_iou.argmax() == idx and (gt_iou >= iou_high_threshold).sum() == 0:
                    is_fallback = True
                    break
            
            assert meets_threshold or is_fallback, f"Bug: anchor {idx} has max IoU {max_iou_for_this_anchor:.3f} < {iou_high_threshold} and is not a fallback"
    neg_inds = torch.where(labels == 0)[0]

    if len(pos_inds) > pos_target:
        perm = torch.randperm(len(pos_inds), device=device)[:pos_target]
        pos_inds = pos_inds[perm]
    if len(neg_inds) > neg_target:
        perm = torch.randperm(len(neg_inds), device=device)[:neg_target]
        neg_inds = neg_inds[perm]

    keep = torch.cat([pos_inds, neg_inds])
    labels = labels[keep]

    best_gt_per_anchor = iou.argmax(dim=1)
    gt_assign = best_gt_per_anchor[keep]
    return keep, labels, gt_assign