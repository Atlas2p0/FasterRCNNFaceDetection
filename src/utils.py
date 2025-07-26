import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import torch
from torchvision.ops import nms

def draw_image_with_box(image_tensor, bounding_boxes_tensor):
    np_image= image_tensor.detach().cpu().numpy()
    np_image= np.transpose(np_image, (1, 2, 0))
    if np_image.max() > 1.0:
        np_image= np_image / 255.0

    np_image= np.clip(np_image, 0.0, 1.0)
    np_image= (np_image * 255).astype(np.uint8)
    image_width, image_height= np_image.shape[:2]

    if not np_image.flags['C_CONTIGUOUS']:
        np_image= np.ascontiguousarray(np_image)
    
    np_boxes= bounding_boxes_tensor.detach().cpu().numpy()
    for box in np_boxes:
        xmin, ymin, xmax, ymax= box
        xmin= int(xmin)
        ymin= int(ymin)
        xmax= int(xmax)
        ymax= int(ymax)
        display_image= cv2.rectangle(np_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    plt.figure(figsize= (10,5))
    plt.axis('off')
    plt.imshow(display_image)

def de_normalize(tensor):
    """Reverses torchvision.transforms.Normalize for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    tensor = tensor.clone() * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

def visualize_anchors_and_gt(image_tensor, anchors, gt_boxes):
    # Convert image tensor to NumPy
    img_to_show = de_normalize(image_tensor).permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img_to_show)

    # Draw a sample of anchor boxes (green)
    # Drawing all 12k+ anchors would freeze the plot
    for anchor in anchors:
        x1, y1, x2, y2 = anchor
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    # Draw ground truth boxes (red)
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.title("Anchors (Green) and Ground Truth (Red)")
    plt.axis("off")
    plt.show()

def decode_predictions(all_anchors, cls_logits, reg_deltas,
                       pre_nms_topk= 1000, post_nms_topk= 300,
                       cls_score_threshold= 0.93, nms_threshold= 0.7):
    scores= torch.sigmoid(cls_logits.squeeze(-1))
    deltas= reg_deltas

    width_anchors= all_anchors[:, 2] - all_anchors[:, 0]
    height_anchors= all_anchors[:, 3] - all_anchors[:, 1]
    x_anchors= all_anchors[:, 0] + width_anchors / 2
    y_anchors= all_anchors[:, 1] + height_anchors / 2

    dx, dy, dw, dh= deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    x_prop= dx * width_anchors + x_anchors
    y_prop= dy * height_anchors + y_anchors
    width_prop= torch.exp(dw) * width_anchors
    height_prop= torch.exp(dh) * height_anchors
    proposals= torch.stack([x_prop - width_prop / 2,
                            y_prop - height_prop / 2,
                            x_prop + width_prop / 2,
                            y_prop + height_prop / 2], dim= 1)
    keep= scores >= cls_score_threshold
    scores= scores[keep]
    proposals= proposals[keep]

    if scores.numel() > pre_nms_topk:
        topk= torch.topk(scores, pre_nms_topk).indices
        scores= scores[topk]
        proposals= proposals[topk]
    keep= nms(proposals, scores, nms_threshold)[:post_nms_topk]
    return scores[keep], proposals[keep]

def bbox_transform(anchors, gt_boxes):
    width_anchors= anchors[:, 2] - anchors[:, 0]
    height_anchors= anchors[:, 3] - anchors[:, 1]
    x_anchors= anchors[:, 0] + width_anchors / 2
    y_anchors= anchors[:, 1] + height_anchors / 2

    width_gts= gt_boxes[:, 2] - gt_boxes[:, 0]
    height_gts= gt_boxes[:, 3] - gt_boxes[:, 1]
    x_gts= gt_boxes[:, 0] + width_gts / 2
    y_gts= gt_boxes[:, 1] + height_gts / 2

    dx= (x_gts - x_anchors) / width_anchors
    dy= (y_gts - y_anchors) / height_anchors
    dw= torch.log(width_gts / width_anchors)
    dh= torch.log(height_gts / height_anchors)

    return torch.stack([dx, dy, dw, dh], dim= 1)

def smooth_l1_loss(pred_reg, targets, beta= 1.0):
    diff= torch.abs(pred_reg - targets)
    loss= torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean(dim= 1)

def decode_deltas(boxes, deltas):
    widths  = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x   = boxes[:, 0] + widths / 2
    ctr_y   = boxes[:, 1] + heights / 2

    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w     = torch.exp(dw) * widths
    pred_h     = torch.exp(dh) * heights

    return torch.stack([
        pred_ctr_x - pred_w / 2,
        pred_ctr_y - pred_h / 2,
        pred_ctr_x + pred_w / 2,
        pred_ctr_y + pred_h / 2
    ], dim=1)