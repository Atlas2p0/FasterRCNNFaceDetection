import numpy as np
import matplotlib.pyplot as plt
import cv2

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

def visualize_anchors_and_gt(image, anchors, gt_boxes):
    # Convert image tensor to NumPy
    np_image = image.detach().cpu().numpy()
    np_image = np.transpose(np_image, (1, 2, 0))  # Convert [C, H, W] to [H, W, C]
    image_width, image_height= np_image.shape[:2]
    if np_image.max() > 1.0:
        np_image= np_image / 255.0

    np_image= np.clip(np_image, 0.0, 1.0)
    np_image = (np_image * 255).astype(np.uint8)  # Scale to [0, 255]
    if not np_image.flags['C_CONTIGUOUS']:
        np_image= np.ascontiguousarray(np_image)
    # Draw anchors
    for anchor in anchors:
        xmin, ymin, xmax, ymax= anchor
        xmin= int(xmin)
        ymin= int(ymin)
        xmax= int(xmax)
        ymax= int(ymax)
        cv2.rectangle(np_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)  # Green for anchors

    # Draw ground truth boxes
    for box in gt_boxes:
        xmin, ymin, xmax, ymax= box
        xmin= int(xmin)
        ymin= int(ymin)
        xmax= int(xmax)
        ymax= int(ymax)
        cv2.rectangle(np_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Red for ground truth

    # Display the image
    plt.figure(figsize=(8, 7))
    plt.imshow(np_image)
    plt.title("Anchors and Ground Truth Boxes")
    plt.axis("off")
    plt.show()

