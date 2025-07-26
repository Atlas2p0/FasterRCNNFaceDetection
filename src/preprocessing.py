import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.ops import nms
import torch.nn as nn
import cv2
import config
from config import data_dir, images_train_dir, images_val_dir, labels_train_dir, labels_val_dir
import os

class FaceDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_list, max_boxes= 30, transform= None):
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
        target_width, target_height= config.IMAGE_SIZE_RESHAPED
        bounding_boxes= []


        if self.transform:
            tensor_image= torch.from_numpy(image).permute(2, 0, 1) / 255.0
            tensor_image= self.transform(tensor_image)
            resized_height, resized_width= tensor_image.shape[1:3]
            with open(label_path, 'r') as f:
                for line in f:
                    parts= line.strip().split()
                    _, cls_name, x1, y1, x2, y2= parts
                    if cls_name.lower() == "human face":
                        continue
                    x1, y1, x2, y2= map(float, (x1, y1, x2, y2))

                    x1= x1 * target_width / original_width
                    y1= y1 * target_height / original_height
                    x2= x2 * target_width / original_width
                    y2= y2 * target_height / original_height
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

def generate_anchor_boxes(image_shape, features, anchor_scales, anchor_ratios):
    num_anchors_per_location= len(anchor_scales) * len(anchor_ratios)
    image_height, image_width= image_shape
    fm_height, fm_width= features.shape[-2:]

    stride_h= image_height / fm_height
    stride_w= image_width / fm_width
    anchor_base_wh= []

    for scale in anchor_scales:
        area= scale * scale
        for ratio in anchor_ratios:
            width= np.sqrt(area * ratio)
            height= np.sqrt(area / ratio)
            anchor_base_wh.append((width, height))
    
    # Get centers in feature map space
    y_feature_map, x_feature_map= np.mgrid[0:fm_height, 0:fm_width]
    centers_x_fm= x_feature_map.ravel()
    centers_y_fm= y_feature_map.ravel()
    
    # Get centers in image space
    centers_x_img= (centers_x_fm + 0.5) * stride_w
    centers_y_img= (centers_y_fm + 0.5) * stride_h
    
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

    x1= np.clip(x1, 0, image_width - 1)
    y1= np.clip(y1, 0, image_height - 1)
    x2= np.clip(x2, 0, image_width - 1)
    y2= np.clip(y2, 0, image_height - 1)
    anchors= np.stack([x1, y1, x2, y2], axis= 1)
    anchors_tensor= torch.tensor(anchors, dtype= torch.float32, device= features.device)
    return anchors_tensor

def apply_deltas_to_boxes(boxes, deltas):
    # Widths and Heights
    widths= boxes[:, 2] - boxes[:, 0]
    heights= boxes[:, 3] - boxes [:, 1]

    # Center Coords
    ctr_x= boxes[:, 0] + 0.5 * widths
    ctr_y= boxes[:, 1] + 0.5 * heights

    dx, dy, dw, dh= deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

    pred_ctr_x= dx * widths + ctr_x
    pred_ctr_y= dy * heights + ctr_y
    pred_w= torch.exp(dw) * widths
    pred_h= torch.exp(dh) * heights

    pred_boxes= torch.stack([
        pred_ctr_x - 0.5 * pred_w,
        pred_ctr_y - 0.5 * pred_h,
        pred_ctr_x + 0.5 * pred_w,
        pred_ctr_y + 0.5 * pred_h
    ], dim= 1)
    
    return pred_boxes

def clamp_boxes_to_img_boundary(boxes, image_shape):
    boxes_x1= boxes[..., 0]
    boxes_y1= boxes[..., 1]
    boxes_x2= boxes[..., 2]
    boxes_y2= boxes[..., 3]

    height, width= image_shape[-2:]
    boxes_x1= boxes_x1.clamp(min= 0, max= width - 1)
    boxes_y1= boxes_y1.clamp(min= 0, max= height - 1)
    boxes_x2= boxes_x2.clamp(min= 0, max= width - 1)
    boxes_y2= boxes_y2.clamp(min= 0, max= height - 1)
    
    boxes= torch.cat( (
        boxes_x1[..., None],
        boxes_y1[..., None],
        boxes_x2[..., None],
        boxes_y2[..., None]
    ), dim= -1)
    return boxes

def filter_proposals(proposals, cls_scores, image_shape, pre_nms_topk, post_nms_topk, nms_thresh= 0.7):
    cls_scores= cls_scores.reshape(-1)
    cls_scores= torch.sigmoid(cls_scores)
    if cls_scores.numel() == 0:
        device = cls_scores.device
        return (torch.empty(0, 4, device=device),
                torch.empty(0, device=device))
    # Pre NMS Filtering
    num_scores= cls_scores.numel()
    k= min(pre_nms_topk, num_scores)
    _, top_n_idx= cls_scores.topk(k)
    cls_scores= cls_scores[top_n_idx]
    proposals= proposals[top_n_idx]

    proposals= clamp_boxes_to_img_boundary(proposals, image_shape)

    keep= nms(proposals, cls_scores, nms_thresh)
    keep= keep[:min(post_nms_topk, keep.numel())]
    return proposals[keep], cls_scores[keep]


def filter_valid_bboxes(gt_boxes):
    valid_mask= (gt_boxes.sum(dim=1) > 0) & ((gt_boxes[:, 2] - gt_boxes[:, 0]) > 0) & ((gt_boxes[:, 3] - gt_boxes[:, 1]) > 0)
    return gt_boxes[valid_mask]

def calculate_iou(boxes1, boxes2, eps= 1e-6):
    # Area of boxes (x2 - x1) * (y2 - y1)
    area1= (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2= (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Get top left x1, y1
    x_left= torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y_top= torch.max(boxes1[:, None, 1], boxes2[:, 1])

    # Get bottom right x2, y2
    x_right= torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y_bottom= torch.min(boxes1[:, None, 3], boxes2[:, 3])

    intersection_area= (x_right - x_left).clamp(min= 0) * (y_bottom - y_top).clamp(min= 0)
    union_area= area1[:, None] + area2 - intersection_area
    return intersection_area / union_area.clamp(min= eps)

def assign_targets_to_anchors(anchors, gt_boxes, iou_high= 0.7, iou_low= 0.3):
    """
    Returns:
        labels         : (A,)  long   -1 ignore, 0 bg, 1 fg
        matched_gt_idx : (A,)  long   index of matched GT (-1 for bg/ignore)
    """
    anchor_boxes, G = anchors.shape[0], gt_boxes.shape[0]
    device = anchors.device

    if G == 0:                       # no GT boxes
        labels = torch.zeros(anchor_boxes, dtype=torch.long, device=device)
        matched_gt_idx = torch.full((anchor_boxes,), -1, dtype=torch.long, device=device)
        return labels, matched_gt_idx

    iou = calculate_iou(gt_boxes, anchors)          # (G, A)

    best_iou, best_gt = iou.max(dim=0)            # (A,)

    # 1. Mark anchors according to IoU thresholds
    labels = torch.full((anchor_boxes,), -1, dtype=torch.long, device=device)
    labels[best_iou < iou_low] = 0                # background
    labels[best_iou >= iou_high] = 1              # foreground

    # 2. Ensure every GT has at least one anchor (even if IoU < high)
    _, best_anchor = iou.max(dim=1)             # (G,)
    labels[best_anchor] = 1

    # 3. Build matched GT indices (-1 for bg/ignore)
    matched_gt_idx = best_gt
    matched_gt_idx[labels == 0] = -1
    matched_gt_idx[labels == -1] = -1

    return labels, matched_gt_idx

def sample_minibatch(labels, pos_ratio= 0.5, num_samples= 256):
    pos_indices= torch.where(labels == 1)[0]
    neg_indices= torch.where(labels == 0)[0]

    num_pos_target= int(num_samples * pos_ratio)
    num_pos_actual= min(len(pos_indices), num_pos_target)

    num_neg_target= num_samples - num_pos_actual
    num_neg_actual= min(len(neg_indices), num_neg_target)

    pos_sampled_indices= pos_indices[torch.randperm(len(pos_indices))[:num_pos_actual]]
    neg_sampled_indices= neg_indices[torch.randperm(len(neg_indices))[:num_neg_actual]]

    keep_indices= torch.cat([pos_sampled_indices, neg_sampled_indices])

    return keep_indices

def create_bbox_deltas(positive_anchors, matched_gt_boxes):
    """
    Calculates the regression targets (tx, ty, tw, th) for RPN training.

    Args:
        positive_anchors (torch.Tensor): Anchors labeled as positive. Shape (P, 4).
        matched_gt_boxes (torch.Tensor): The GT boxes matched to each positive anchor. Shape (P, 4).

    Returns:
        torch.Tensor: The calculated regression targets. Shape (P, 4).
    """
    # Ensure boxes are in [x1, y1, x2, y2] format
    # Convert to [center_x, center_y, width, height]
    anchor_widths = positive_anchors[:, 2] - positive_anchors[:, 0]
    anchor_heights = positive_anchors[:, 3] - positive_anchors[:, 1]
    anchor_ctr_x = positive_anchors[:, 0] + 0.5 * anchor_widths
    anchor_ctr_y = positive_anchors[:, 1] + 0.5 * anchor_heights

    gt_widths = matched_gt_boxes[:, 2] - matched_gt_boxes[:, 0]
    gt_heights = matched_gt_boxes[:, 3] - matched_gt_boxes[:, 1]
    gt_ctr_x = matched_gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = matched_gt_boxes[:, 1] + 0.5 * gt_heights

    # Avoid division by zero
    eps = torch.finfo(anchor_widths.dtype).eps
    anchor_widths = anchor_widths.clamp(min=eps)
    anchor_heights = anchor_heights.clamp(min=eps)

    # Calculate targets as per the Faster R-CNN paper
    tx = (gt_ctr_x - anchor_ctr_x) / anchor_widths
    ty = (gt_ctr_y - anchor_ctr_y) / anchor_heights
    tw = torch.log(gt_widths / anchor_widths)
    th = torch.log(gt_heights / anchor_heights)

    return torch.stack([tx, ty, tw, th], dim=1)