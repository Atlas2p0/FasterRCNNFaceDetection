import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import torch
from torchvision.ops import nms
from preprocessing import apply_deltas_to_boxes, clamp_boxes_to_img_boundary
import torch.nn.functional as F

def draw_image_with_box(image_tensor, bounding_boxes_tensor):
    """
    Draws bounding boxes on an image and displays the result.
    
    This function takes a tensor representing an image and a tensor of bounding boxes,
    converts them to appropriate formats, draws the bounding boxes on the image, and
    displays the result using matplotlib. The function handles tensor-to-numpy conversion,
    normalization, and ensures the image is in the correct format for OpenCV operations.
    
    Args:
        image_tensor (torch.Tensor): Tensor representing the image with shape (C, H, W)
                                   where C is the number of channels (typically 3 for RGB).
        bounding_boxes_tensor (torch.Tensor): Tensor of bounding boxes with shape (N, 4)
                                            where each row is (xmin, ymin, xmax, ymax).
    
    Returns:
        None: The function displays the image with bounding boxes using matplotlib.
    """
    # Convert tensor image to numpy and transpose dims from (C, H, W) to (H, W, C) for visualization
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
    """
    Visualizes anchor boxes and ground truth boxes on an image.
    
    This function creates a visualization that overlays anchor boxes (in green) and 
    ground truth boxes (in red) on an input image. This is useful for understanding 
    the anchor box generation process and how they relate to ground truth annotations 
    in object detection models.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor with shape (C, H, W) in normalized form.
        anchors (torch.Tensor or numpy.ndarray): Anchor boxes with shape (N, 4) where each row
                                              represents (x1, y1, x2, y2) coordinates.
        gt_boxes (torch.Tensor or numpy.ndarray): Ground truth boxes with shape (M, 4) where each row
                                                represents (x1, y1, x2, y2) coordinates.
    
    Returns:
        None: The function displays the visualization using matplotlib.
    """
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

    plt.title("Anchors/Proposals (Green) and Ground Truth (Red)")
    plt.axis("off")
    plt.show()



@torch.no_grad()
def roi_head_inference(features, proposals, image_shape, roi_head, score_thresh= 0.5, nms_thresh= 0.2):
    """
    Performs inference using the Region of Interest (ROI) head to generate final object detections.
    
    This function processes region proposals through the ROI head to obtain class predictions and
    bounding box refinements. It then applies score thresholding and Non-Maximum Suppression (NMS)
    to produce the final set of detections. The function is a key component in two-stage object
    detectors like Faster R-CNN, where it refines proposals from the Region Proposal Network (RPN).
    
    Args:
        features (torch.Tensor): Feature map tensor with shape (batch, channels, height, width)
                                extracted by the backbone network.
        proposals (torch.Tensor): Tensor of region proposals with shape (N, 4) where each row
                                 represents (x1, y1, x2, y2) coordinates.
        image_shape (tuple): Shape of the input image as (height, width).
        roi_head (nn.Module): The ROI head module that processes proposals to generate class
                             predictions and bounding box refinements.
        score_thresh (float, optional): Confidence threshold for filtering detections. Only
                                      detections with scores above this threshold are kept.
                                      Default is 0.5.
        nms_thresh (float, optional): IoU threshold for Non-Maximum Suppression. Detections
                                     with IoU above this threshold are considered duplicates
                                     and only the one with the highest score is kept.
                                     Default is 0.1.
    
    Returns:
        tuple: A tuple containing:
            - pred_boxes (torch.Tensor): Final bounding box predictions with shape (M, 4)
                                        where M is the number of detections after filtering.
            - scores (torch.Tensor): Confidence scores for the final detections with shape (M,).
    """

    # Use RoI to predict cls_logits and bounding box deltas
    roi_out= roi_head(features, [proposals], [image_shape], gt_boxes= None)
    cls_logits= roi_out['cls_logits']
    bbox_deltas= roi_out['bbox_deltas']

    # Print range of class logits for debugging/monitoring purposes
    # This helps understand the distribution of raw_scores
    print("cls_logits range:", cls_logits.min().item(), cls_logits.max().item())
    # Apply softmax to convert logits to probabilities
    scores= F.softmax(cls_logits, dim= 1)[:, 1]

    print(scores.max())
    # Apply deltas to proposals and ensure they are within image boundaries
    pred_boxes= apply_deltas_to_boxes(proposals, bbox_deltas)
    pred_boxes= clamp_boxes_to_img_boundary(pred_boxes, image_shape)

    # Filter proposals based on score threshold
    keep= scores > score_thresh
    pred_boxes= pred_boxes[keep]
    scores= scores[keep]
    # Apply Non-Maximum Suppression to remove overlapping detections
    keep= nms(pred_boxes, scores, nms_thresh)
    return pred_boxes[keep], scores[keep]

@torch.no_grad()
def generate_proposals(images, image_shapes, backbone, rpn_model):
    """
    Generates region proposals using a backbone network and Region Proposal Network (RPN).
    
    This function processes input images through a backbone network to extract features,
    then passes these features to an RPN model to generate region proposals. The function
    uses mixed precision training (autocast) for improved performance and efficiency.
    The proposals are truncated to the top 512 for each image to limit the number of
    proposals passed to subsequent stages.
    
    Args:
        images (torch.Tensor): Batch of input images with shape (batch_size, channels, height, width).
        image_shapes (list): List of image shapes as (height, width) tuples for each image in the batch.
        backbone (nn.Module): Backbone neural network (e.g., ResNet, VGG) that extracts features
                             from the input images.
        rpn_model (nn.Module): Region Proposal Network that takes features and generates
                              object proposals.
    
    Returns:
        tuple: A tuple containing:
            - proposals (list): List of proposal tensors, one per image. Each tensor has shape
                               (num_proposals, 4) where num_proposals is at most 512, and each
                               row represents (x1, y1, x2, y2) coordinates.
            - features (torch.Tensor): Feature maps extracted by the backbone with shape
                                      (batch_size, channels, feature_height, feature_width).
    
    Note:
        The function uses torch.cuda.amp.autocast() for automatic mixed precision, which
        can improve performance and reduce memory usage on compatible GPUs.
    """
    features= backbone(images)
    with torch.cuda.amp.autocast():
      rpn_out= rpn_model(feat= features, image_shapes= image_shapes, gt_boxes= None)
    proposals= [p[:512] for p in rpn_out['proposals']]
    
    return proposals, features