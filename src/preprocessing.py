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
    """
    A PyTorch Dataset class for loading face images and their corresponding bounding box annotations.
    
    This dataset loads images and their associated bounding box labels, processes them,
    and returns them in a format suitable for training object detection models like Faster R-CNN.
    
    Args:
        image_dir (str): Path to the directory containing images.
        label_dir (str): Path to the directory containing label files.
        image_list (list): List of image filenames to include in the dataset.
        max_boxes (int, optional): Maximum number of bounding boxes per image. 
                                 Default is 30.
        transform (callable, optional): Optional transform to be applied on images.
                                       Default is None.
    
    Attributes:
        image_dir (str): Path to the directory containing images.
        label_dir (str): Path to the directory containing label files.
        image_list (list): List of image filenames to include in the dataset.
        max_boxes (int): Maximum number of bounding boxes per image.
        transform (callable, optional): Transform to be applied on images.
    
    Methods:
        __len__: Returns the number of images in the dataset.
        __getitem__: Loads and processes a single image and its annotations.
    """
    def __init__(self, image_dir, label_dir, image_list, max_boxes= 30, transform= None):
        self.image_dir= image_dir
        self.label_dir= label_dir
        self.image_list= image_list
        self.transform= transform
        self.max_boxes= max_boxes

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        # Method  that retrieves and processes a single item from the dataset

        # Get image and label file names from their respective dirs
        image_name= self.image_list[idx]
        image_path= os.path.join(self.image_dir, image_name)
        label_path= os.path.join(self.label_dir, image_name.rsplit('.', 1)[0] + '.txt')

        # Image loading
        image= cv2.imread(image_path)
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Get original image height and width
        original_height, original_width= image.shape[:2]
        # Get target width and height from config file
        target_width, target_height= config.IMAGE_SIZE_RESHAPED
        # Initialize list of ground truth boxes
        bounding_boxes= []

        # Check if transform is specified
        if self.transform:
            # Convert numpy image to tensor
            tensor_image= torch.from_numpy(image).permute(2, 0, 1) / 255.0
            tensor_image= self.transform(tensor_image)
            
            # Get bounding boxes and transform them to fit the new image size correctly
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
        # Get number of bboxes for the current image            
        num_curr_boxes= len(bounding_boxes)

        # Pad if current boxes < max_boxes specified
        if num_curr_boxes < self.max_boxes:
            padding= np.zeros((self.max_boxes - num_curr_boxes, 4), dtype= np.float32)
            padded_boxes= np.concatenate((np.array(bounding_boxes, dtype= np.float32),
                                           padding), axis= 0)
        # Truncate
        else:
            padded_boxes= np.array(bounding_boxes[:self.max_boxes], dtype=np.float32)
        
        boxes_tensor= torch.tensor(padded_boxes, dtype= torch.float32)

        return {
            'image': tensor_image,
            'boxes': boxes_tensor
        }

def generate_anchor_boxes(image_shape, features, anchor_scales, anchor_ratios):
    """
    Generates anchor boxes for a feature map in an object detection model.
    
    This function creates anchor boxes of different scales and aspect ratios at every position
    in a feature map. These anchors serve as reference bounding boxes for object detection.
    
    Args:
        image_shape (tuple): Shape of the input image as (height, width).
        features (torch.Tensor): Feature map tensor with shape (batch, channels, height, width).
        anchor_scales (list): List of scales for anchor boxes (e.g., [8, 16, 32]).
        anchor_ratios (list): List of aspect ratios for anchor boxes (e.g., [0.5, 1, 2]).
    
    Returns:
        torch.Tensor: Tensor of anchor boxes with shape (num_anchors, 4) where each row
                     represents (x1, y1, x2, y2) coordinates of an anchor box.
    """
    # Calculate number of anchor boxes per feature map location
    num_anchors_per_location= len(anchor_scales) * len(anchor_ratios)
    # Unpack image dims
    image_height, image_width= image_shape
    # Get height and width of the feature map
    fm_height, fm_width= features.shape[-2:]
    # Calculate stride (number of pixels in the original that correspond to one cell in the feature map)
    # This is essentially the receptive field of each feature map cell
    stride_h= image_height / fm_height
    stride_w= image_width / fm_width
    # Initialize an empty list to store base anchor widths and heights
    anchor_base_wh= []

    for scale in anchor_scales:
        # For every anchor scale calculate the area of the anchor box
        area= scale * scale
        for ratio in anchor_ratios:
            # For every apsect ratio get the width and height for the anchor box at this aspect ratio
            width= np.sqrt(area * ratio)
            height= np.sqrt(area / ratio)
            # Store in anchor bases list
            anchor_base_wh.append((width, height))
    
    # Create a grid of coords for the feature map
    y_feature_map, x_feature_map= np.mgrid[0:fm_height, 0:fm_width]

    # Get centers in feature map space
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

    # Clip to image boundaries
    x1= np.clip(x1, 0, image_width - 1)
    y1= np.clip(y1, 0, image_height - 1)
    x2= np.clip(x2, 0, image_width - 1)
    y2= np.clip(y2, 0, image_height - 1)

    anchors= np.stack([x1, y1, x2, y2], axis= 1)
    anchors_tensor= torch.tensor(anchors, dtype= torch.float32, device= features.device)

    return anchors_tensor

def apply_deltas_to_boxes(boxes, deltas):
    """
    Applies predicted deltas to anchor boxes to generate refined bounding boxes.
    
    This function is a key component in object detection models like Faster R-CNN,
    where it transforms anchor boxes using predicted adjustments (deltas) to better
    match the ground truth objects. The deltas represent adjustments to the center
    coordinates and dimensions of the boxes.
    
    Args:
        boxes (torch.Tensor): Tensor of anchor boxes with shape (N, 4) where each row
                             represents (x1, y1, x2, y2) coordinates.
        deltas (torch.Tensor): Tensor of predicted deltas with shape (N, 4) where each row
                              represents (dx, dy, dw, dh) adjustments for the boxes.
    
    Returns:
        torch.Tensor: Tensor of refined bounding boxes with shape (N, 4) where each row
                     represents (x1, y1, x2, y2) coordinates after applying the deltas.
    """
    # Widths and Heights
    widths= boxes[:, 2] - boxes[:, 0]
    heights= boxes[:, 3] - boxes [:, 1]

    # Center Coords
    ctr_x= boxes[:, 0] + 0.5 * widths
    ctr_y= boxes[:, 1] + 0.5 * heights

    # Get deltas
    dx, dy, dw, dh= deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    # Calculate the predicted center x|y-coordinate by:
    # 1. Multiplying the x|y-delta by the original height
    # 2. Adding the result to the original center x|y-coordinate
    pred_ctr_x= dx * widths + ctr_x
    pred_ctr_y= dy * heights + ctr_y
    # Calculate the predicted height|width by:
    # 1. Exponentiating the height|width delta (to convert from log-space)
    # 2. Multiplying by the original height|width
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
    """
    Clamps bounding box coordinates to ensure they are within image boundaries.
    
    This function ensures that all bounding boxes are fully contained within the image
    by clamping their coordinates to the image dimensions. This is important for
    preventing invalid box coordinates that extend beyond the image boundaries.
    
    Args:
        boxes (torch.Tensor): Tensor of bounding boxes with shape (..., 4) where each box
                            is represented as (x1, y1, x2, y2) coordinates.
        image_shape (tuple): Shape of the image as (..., height, width). The function uses
                            the last two dimensions to determine the image boundaries.
    
    Returns:
        torch.Tensor: Tensor of clamped bounding boxes with the same shape as input,
                    where all coordinates are within the image boundaries.
    """
    # Extract x1, y1, x2, y2 coordinates of all boxes
    boxes_x1= boxes[..., 0]
    boxes_y1= boxes[..., 1]
    boxes_x2= boxes[..., 2]
    boxes_y2= boxes[..., 3]

    # Get height and width of the image (the boundaries)
    height, width= image_shape[-2:]

    # Clamp box coords to be inside the image
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
    """
    Filters object proposals using classification scores and Non-Maximum Suppression (NMS).
    
    This function processes raw object proposals by:
    1. Selecting the top-k proposals based on classification scores (pre-NMS)
    2. Clamping proposal boxes to image boundaries
    3. Applying NMS to remove overlapping proposals
    4. Selecting the top-k proposals after NMS
    
    Args:
        proposals (torch.Tensor): Tensor of bounding box proposals with shape (N, 4) 
                                where each row is (x1, y1, x2, y2).
        cls_scores (torch.Tensor): Tensor of classification scores with shape (N,) or (N, 1).
        image_shape (tuple): Shape of the image as (..., height, width).
        pre_nms_topk (int): Number of top proposals to keep before applying NMS.
        post_nms_topk (int): Number of top proposals to keep after applying NMS.
        nms_thresh (float, optional): IoU threshold for NMS. Default is 0.7.
    
    Returns:
        tuple: A tuple containing:
            - filtered_proposals (torch.Tensor): Filtered bounding boxes with shape (M, 4) 
                                                where M ≤ post_nms_topk.
            - filtered_scores (torch.Tensor): Corresponding classification scores with shape (M,).
    """
    # Reshape classification scores to a 1D tensor
    # Ensures consistent shape regardless of input(N, ) or (N, 1)
    cls_scores= cls_scores.reshape(-1)
    cls_scores= torch.sigmoid(cls_scores)

    # If no scores (empty tensor) return empty proposals and scores
    if cls_scores.numel() == 0:
        device = cls_scores.device
        return (torch.empty(0, 4, device=device),
                torch.empty(0, device=device))
    
    # Pre NMS Filtering

    # Get Total number of scores
    num_scores= cls_scores.numel()
    # Determine how many top proposals to keep before NMS
    k= min(pre_nms_topk, num_scores)
    # Get indices of the top k scores (largest values)
    _, top_n_idx= cls_scores.topk(k)
    # Select top k scores using the indices
    cls_scores= cls_scores[top_n_idx]
    # Select corresponding proposals using the same indices
    proposals= proposals[top_n_idx]

    # Ensure all proposals are within image boundaries 
    proposals= clamp_boxes_to_img_boundary(proposals, image_shape)

    # Apply Non-Maximum Suppression (NMS) to remove overlapping proposals
    # NMS returns indices of proposals to keep, based on:
    # 1. Sorting proposals by score (descending)
    # 2. Keeping the highest scoring proposal
    # 3. Removing proposals with IoU > nms_thresh with the kept proposal
    # 4. Repeating until no proposals remain
    keep= nms(proposals, cls_scores, nms_thresh)
    keep= keep[:min(post_nms_topk, keep.numel())]

    return proposals[keep], cls_scores[keep]


def filter_valid_bboxes(gt_boxes):
    valid_mask= (gt_boxes.sum(dim=1) > 0) & ((gt_boxes[:, 2] - gt_boxes[:, 0]) > 0) & ((gt_boxes[:, 3] - gt_boxes[:, 1]) > 0)
    return gt_boxes[valid_mask]

def calculate_iou(boxes1, boxes2, eps= 1e-6):
    """
    Calculate Intersection over Union (IoU) between two sets of bounding boxes.
    
    This function computes the IoU for each pair of boxes from two sets. IoU is a measure
    of overlap between two bounding boxes, calculated as the area of intersection divided
    by the area of union. The function handles broadcasting to compute pairwise IoU between
    all boxes in boxes1 and all boxes in boxes2.
    
    Args:
        boxes1 (torch.Tensor): Tensor of bounding boxes with shape (N, 4) where each row
                              represents (x1, y1, x2, y2) coordinates.
        boxes2 (torch.Tensor): Tensor of bounding boxes with shape (M, 4) where each row
                              represents (x1, y1, x2, y2) coordinates.
        eps (float, optional): Small epsilon value to prevent division by zero. Default is 1e-6.
    
    Returns:
        torch.Tensor: IoU matrix with shape (N, M) where element [i, j] represents the
                     IoU between boxes1[i] and boxes2[j].
    """
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
    Assign target labels and matched ground truth boxes to anchor boxes based on IoU thresholds.
    
    This function implements the anchor assignment strategy used in object detection models like Faster R-CNN.
    It classifies each anchor as foreground (positive), background (negative), or ignore based on its IoU 
    with ground truth boxes. It also identifies which ground truth box each anchor is matched to.
    
    Args:
        anchors (torch.Tensor): Tensor of anchor boxes with shape (A, 4) where each row
                               represents (x1, y1, x2, y2) coordinates.
        gt_boxes (torch.Tensor): Tensor of ground truth boxes with shape (G, 4) where each row
                                represents (x1, y1, x2, y2) coordinates.
        iou_high (float, optional): IoU threshold for foreground assignment. Anchors with IoU >= this
                                    value are considered foreground. Default is 0.7.
        iou_low (float, optional): IoU threshold for background assignment. Anchors with IoU < this
                                   value are considered background. Default is 0.3.
    
    Returns:
        tuple: A tuple containing:
            - labels (torch.Tensor): Tensor of shape (A,) with classification labels:
                * 1 for foreground anchors (positive samples)
                * 0 for background anchors (negative samples)
                * -1 for ignored anchors (neither foreground nor background)
            - matched_gt_idx (torch.Tensor): Tensor of shape (A,) with indices of matched ground truth
                                           boxes for each anchor. Value is -1 for background and
                                           ignored anchors.
    
    Notes:
        - If no ground truth boxes are provided (G=0), all anchors are assigned as background.
        - Each ground truth box is guaranteed to be matched to at least one anchor (the one with
          highest IoU), even if its IoU is below iou_high.
        - Anchors with IoU between iou_low and iou_high that are not explicitly matched to a
          ground truth box are marked as ignored (-1).
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
    """
    Samples a minibatch of anchors with balanced positive and negative examples.
    
    This function is used during training of the Region Proposal Network to select a subset of anchors for computing the loss.
    It aims to maintain a specified ratio of positive (foreground) to negative (background) samples
    to prevent class imbalance. The function randomly samples from available positive and negative
    anchors to create a minibatch of the desired size.
    
    Args:
        labels (torch.Tensor): Tensor of shape (A,) containing anchor labels:
                              - 1 for foreground (positive) anchors
                              - 0 for background (negative) anchors
                              - -1 for ignored anchors (not considered for sampling)
        pos_ratio (float, optional): Desired ratio of positive samples in the minibatch.
                                    Default is 0.5.
        num_samples (int, optional): Total number of samples to select for the minibatch.
                                    Default is 256.
    
    Returns:
        torch.Tensor: Tensor of indices representing the selected anchors for the minibatch.
                     The indices correspond to positions in the original labels tensor.
    """
    # Unpack all positive and negative indices into their respective vars
    pos_indices= torch.where(labels == 1)[0]
    neg_indices= torch.where(labels == 0)[0]

    # Get actual number of positive anchors and target pos anchors
    num_pos_target= int(num_samples * pos_ratio)
    num_pos_actual= min(len(pos_indices), num_pos_target)
    # Get actual number of neg anchors and target neg anchors
    num_neg_target= num_samples - num_pos_actual
    num_neg_actual= min(len(neg_indices), num_neg_target)

    # Randomly sample positive|negative anchors:
    # 1. torch.randperm(len(pos_indices)) generates a random permutation of indices for positive|negative anchors
    # 2. [:num_pos_actual] selects the first num_pos_actual indices from this permutation
    # 3. pos_indices[...] uses these indices to select the actual positive|negative anchor indices
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