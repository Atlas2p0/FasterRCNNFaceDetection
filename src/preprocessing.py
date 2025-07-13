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
    def __init__(self, image_dir, label_dir, image_list, max_boxes= 5, transform= None):
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
                    box= list([x1 / resized_width, y1 / resized_height, x2 / resized_width, y2 / resized_height])
                    bounding_boxes.append(box)
        else:
            tensor_image= torch.from_numpy(image).permute(2, 0, 1).float()
            with open(label_path, 'r') as f:
                for line in f:
                    _, x, y, box_width, box_height= map(float, line.strip().split())
                    x1= int((x - box_width / 2) * original_width)
                    y1= int((y - box_height / 2) * original_height)
                    x2= int((x + box_width / 2) * original_width)
                    y2= int((y + box_height / 2) * original_height)
                    box= list([x1, y1, x2, y2])
                    bounding_boxes.append(box)

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

def generate_anchor_boxes(feature_map_shape, image_shape, scales, ratios, stride):

    feature_map_height, feature_map_width= feature_map_shape
    image_height, image_width= image_shape

    base_anchors_wh= []
    for scale in scales:
        for ratio in ratios:
            width= scale * np.sqrt(ratio)
            height= scale / np.sqrt(ratio)
            base_anchors_wh.append([width, height])
        
    base_anchors_wh= np.array(base_anchors_wh)
    num_base_anchors= base_anchors_wh.shape[0]

    

    center_x_coords= np.arange(feature_map_width) * stride + stride / 2.0
    center_y_coords= np.arange(feature_map_height) * stride + stride / 2.0


    center_y_grid, center_x_grid= np.meshgrid(center_y_coords, center_x_coords)
    centers_xy= np.stack([center_x_grid, center_y_grid], axis= -1) # Shape [16, 16, 2]
    if center_x_coords[-1] + stride / 2.0 < image_width:
        center_x_coords= np.append(center_x_coords, image_width - stride / 2.0)
    if center_y_coords[-1] + stride / 2.0 < image_height:
        center_y_coords= np.append(center_y_coords, image_height - stride / 2.0)

        
    centers_expanded= np.expand_dims(centers_xy, axis= 2) # Shape [16, 16, 1, 2]
    base_anchors_expanded= np.expand_dims(base_anchors_wh, axis= (0, 1)) # Shape [1, 1, 9, 2]
    grid_height, grid_width= centers_xy.shape[:2]
    all_anchors_xyxy_relative= np.zeros((grid_height, grid_width, num_base_anchors, 4), dtype= np.float32)

    all_anchors_xyxy_relative[:, :, :, 0]= centers_expanded[:, :, :, 0] - base_anchors_expanded[:, :, :, 0] / 2.0
    all_anchors_xyxy_relative[:, :, :, 1]= centers_expanded[:, :, :, 1] - base_anchors_expanded[:, :, :, 1] / 2.0

    all_anchors_xyxy_relative[:, :, :, 2]= centers_expanded[:, :, :, 0] + base_anchors_expanded[:, :, :, 0] / 2.0
    all_anchors_xyxy_relative[:, :, :, 3]= centers_expanded[:, :, :, 1] + base_anchors_expanded[:, :, :, 1] / 2.0

    all_anchors_xyxy_relative[:, :, :, 0]= np.clip(all_anchors_xyxy_relative[:, :, :, 0], 0, image_width - 1)
    all_anchors_xyxy_relative[:, :, :, 1]= np.clip(all_anchors_xyxy_relative[:, :, :, 1], 0, image_height - 1)
    all_anchors_xyxy_relative[:, :, :, 2]= np.clip(all_anchors_xyxy_relative[:, :, :, 2], 0, image_width - 1)
    all_anchors_xyxy_relative[:, :, :, 3]= np.clip(all_anchors_xyxy_relative[:, :, :, 3], 0, image_height - 1)

    all_anchors_xyxy_relative[:, :, :, 0]/= image_width
    all_anchors_xyxy_relative[:, :, :, 1]/= image_height
    all_anchors_xyxy_relative[:, :, :, 2]/= image_width
    all_anchors_xyxy_relative[:, :, :, 3]/= image_height

    all_anchors_xyxy= all_anchors_xyxy_relative.reshape(-1, 4)
    all_anchors_xyxy.shape, all_anchors_xyxy_relative.shape

    return torch.tensor(all_anchors_xyxy, dtype= torch.float32)


def filter_valid_bboxes(gt_boxes):
    valid_mask= (gt_boxes.sum(dim= 1) > 0)
    return gt_boxes[valid_mask]

def calculate_iou(anchor_boxes, gt_boxes):
    anchor_boxes= anchor_boxes.unsqueeze(1) # Shape: [num_anchors, 1, 4]
    gt_boxes= gt_boxes.unsqueeze(0) # Shape: [1, num_gt_boxes, 4]

    inter_xmin= torch.max(anchor_boxes[:, :, 0], gt_boxes[:, :, 0])
    inter_ymin= torch.max(anchor_boxes[:, :, 1], gt_boxes[:, :, 1])
    inter_xmax= torch.min(anchor_boxes[:, :, 2], gt_boxes[:, :, 2])
    inter_ymax= torch.min(anchor_boxes[:, :, 3], gt_boxes[:, :, 3])

    inter_width= torch.clamp(inter_xmax - inter_xmin, min= 0)
    inter_height= torch.clamp(inter_ymax - inter_ymin, min= 0)
    inter_area= inter_width * inter_height

    anchor_area= (anchor_boxes[:, :, 2] - anchor_boxes[:, :, 0]) * (anchor_boxes[:, :, 3] - anchor_boxes[:, :, 1])
    gt_area= (gt_boxes[:, :, 2] - gt_boxes[:, :, 0]) * (gt_boxes[:, :, 3] - gt_boxes[:, :, 1])
    union_area= anchor_area + gt_area - inter_area

    iou= inter_area / (union_area + 1e-6)
    return iou

def match_anchors_to_gt(anchor_boxes, gt_boxes, iou_high_threshold= 0.6, iou_low_threshold= 0.3, device= 'cpu',
                        num_samples= 32, positive_ratio= 0.25):
    gt_boxes= filter_valid_bboxes(gt_boxes)
    num_anchors= anchor_boxes.shape[0]
    if gt_boxes.numel() == 0:
        labels= torch.zeros((num_anchors, ), dtype= torch.int32, device= device)
        regression_targets= torch.zeros((num_anchors,), dtype= torch.float32, device= device)
        return {"labels": labels, "regression_targets": regression_targets}

    anchor_boxes= anchor_boxes.to(device)
    gt_boxes= gt_boxes.to(device)

    iou_matrix= calculate_iou(anchor_boxes, gt_boxes)
    labels= torch.full((num_anchors, ), -1, dtype= torch.int32, device= device)
    max_iou, max_iou_indices= iou_matrix.max(dim= 1)
    
    labels[max_iou >= iou_high_threshold]= 1
    labels[max_iou < iou_low_threshold]= 0
    for i in range(gt_boxes.shape[0]):
        best_anchor_idx= iou_matrix[:, i].argmax()
        labels[best_anchor_idx]= 1
    
    positive_indices= torch.where(labels == 1)[0]
    negative_indices= torch.where(labels == 0)[0]
    num_positive= int(num_samples * positive_ratio)
    num_negative= num_samples - num_positive

    if positive_indices.numel() > num_positive:
        positive_indices= positive_indices[torch.randperm(positive_indices.numel())[:num_positive]]
    if negative_indices.numel() > num_negative:
        negative_indices= negative_indices[torch.randperm(negative_indices.numel())[:num_negative]]
    sampled_indices= torch.cat([positive_indices, negative_indices])
    labels= torch.full((num_anchors,), -1, dtype= torch.int32, device= device)
    labels[sampled_indices]= 0
    labels[positive_indices]= 1

    regression_targets= torch.zeros((num_anchors, 4), dtype= torch.float32, device= device)
    if positive_indices.numel() > 0:
        positive_anchors= anchor_boxes[positive_indices]
        corresponding_gt_boxes= gt_boxes[max_iou_indices[positive_indices]]
        
        anchor_widths= positive_anchors[:, 2] - positive_anchors[:, 0]
        anchor_heights= positive_anchors[:, 3] - positive_anchors[:, 1]
        anchor_centers_x= positive_anchors[:, 0] + 0.5 * anchor_widths
        anchor_centers_y= positive_anchors[:, 1] + 0.5 * anchor_heights

        gt_widths= corresponding_gt_boxes[:, 2] - corresponding_gt_boxes[:, 0]
        gt_heights= corresponding_gt_boxes[:, 3] - corresponding_gt_boxes[:, 1]
        gt_centers_x= corresponding_gt_boxes[:, 0] + 0.5 * gt_widths
        gt_centers_y= corresponding_gt_boxes[:, 1] + 0.5 * gt_heights

        regression_targets[positive_indices, 0]= (gt_centers_x - anchor_centers_x) / anchor_widths
        regression_targets[positive_indices, 1]= (gt_centers_y - anchor_centers_y) / anchor_heights

        regression_targets[positive_indices, 2]= torch.log(gt_widths / anchor_widths)
        regression_targets[positive_indices, 3]= torch.log(gt_heights / anchor_heights)

    return  {"labels": labels, "regression_targets": regression_targets}