# Project overview
This project implements Faster R-CNN for face detection from first principles, including:
* Region Proposal Network (RPN) for generating object proposals
* Region of Interest (ROI) head for classifying and refining proposals
* Asynchronous video processing pipeline for real-time detection
* Custom dataset for handling training and inference

 The implementation focuses on clarity and educational value while maintaining performance close to real-time applications.

 # Key Features
 * **Complete Faster R-CNN Implementation**: All key components are built from scratch without relying on prebuilt detection frameworks
 * **Asynchronous Processing**: Efficient video processing pipeline that separates frame reading from model inference
 * **Real-time performance**: Optimized for real-time object detection with performance monitoring
 * **Visualization Tools**: Built-in utilities for visualizing anchors, groundtruth boxes, and detection results
 * **Dynamic Batch Sizing**: Automatically adjusts batch size based on processing time to maintain target FPS
 * **Modular Design**: Clean separation of components for easy understanding and extension


# Project Showcase

## Real-time Object Detection
The Implementation processes video in real-time, displaying bounding boxes and confidence scores for detected objects. The system maintains a target FPS through dynamic batch sizing and asynchronous processing.


https://github.com/user-attachments/assets/46ce1832-1244-40d4-add1-9e965627a5ea


## Anchor Visualization
Visualization of anchor boxes (green) overlaid on ground truth boxes (red) to demonstrate the anchor generation process and matching strategy.
<img width="981" height="814" alt="image" src="https://github.com/user-attachments/assets/476e17aa-b032-4494-8589-4615ba578740" />
<img width="910" height="811" alt="image" src="https://github.com/user-attachments/assets/edc2b780-4e0f-4beb-b2c7-1cab289d25ac" />


# Performance Metrics

## Region Proposal Network (RPN) Performance
| Dataset       | Total Loss | Objectness Loss | Regression Loss | Recall@0.5 IoU |
|---------------|------------|-----------------|-----------------|----------------|
| Training      | 0.0154     | 0.0147          | 0.0006          | -              |
| Validation    | 0.0265     | 0.0253          | 0.0012          | 83%            |
| Test (Unseen) | 0.0293     | 0.0279          | 0.0014          | 80%            |
<img width="1185" height="917" alt="image" src="https://github.com/user-attachments/assets/15d3dd20-52cc-471b-a9c5-9f5fb5cd367c" />

## Region of Interest (ROI) Head Performance
| Dataset       | Total Loss | Classification Loss | Regression Loss | Precision | Recall@0.4 IoU | mAP   |
|---------------|------------|---------------------|-----------------|-----------|----------------|-------|
| Training      | 0.2111     | 0.2077              | 0.0035          | -         | -              | -     |
| Validation    | 0.2228     | 0.2192              | 0.0036          | 95.1%     | 46.5%          | 76.3% |
| Test (Unseen) | 0.2199     | 0.2164              | 0.0036          | 90.8%     | 45.2%          | 74.6% |
<img width="1171" height="825" alt="image" src="https://github.com/user-attachments/assets/8c19e939-1f3d-4503-b002-66e278e19cf7" />

## Real-Time Inference Speed
| Hardware       | FPS  |
|----------------|------|
| CPU (R5 7600x) | 2-3  |
| GPU (RTX 3060) | 16   |

# Key Algorithms
* **Anchor Generation**: Creates anchor boxes at different scales and aspect ratios for each position in the feature map
* **Target Assignment**: Assigns anchors to ground truth boxes based on Intersection over Union (IoU) thresholds
* **Bounding Box Regression**: Applies predicted deltas to refine proposal coordinates
* **Minibatch Sampling**: Balances positive and negative samples for training stability

# Training/Validation and Real-Time Performance Metrics

* **Anchor Classification Loss**: Binary cross-entropy loss measuring how well the RPN distinguishes between foreground (faces) anchors and background anchors
* **Anchor Regression Loss**: Smooth L1 loss measuring the accuracy of bounding box adjustments for foreground anchors
* **Total RPN Loss**: Weighted sum of classification and regression losses
* **Recall@0.5 IoU (RPN)**: Proportion of ground truth objects that have an IoU > 0.5 with at least one of the top K proposals
* **Recall@0.4 IoU (ROI Head)**: Measures the model's ability to detect all actual objects
* **Precision (ROI Head)**: The accuracy of positive predictions made by the RoI head
* **Mean Average Precision (mAP) (ROI Head)**: Considered to be the primary metric for evaluating object detection models. It combines precision and recall into a single value that represents the model's overall performance across differenct confidence thresholds

1. Region Proposal Network (RPN)
    * Achieved total loss of 0.0154 (0.0147 objectness loss + 0.0006 regression loss) on training data, and total loss of 0.0265 (0.0253 objectness loss + 0.012 regression loss) on validation data.
    * Achieved total loss of 0.0293 (0.0279 objectness loss + 0.0014 regression loss) on new unseen test data
    * Achieved 83% recall@0.5 IoU on validation data and 80% recall @0.5IoU on new unseen test data
2. Region of Interest head (RoI)
    * Achieved total loss of 0.2111 (0.2077 objectness loss + 0.0035 regression loss) on train data, and total loss of 0.2228 (0.2192 objectness loss + 0.0036 regression loss) on validation data
    * Achieved Precision of 95.1%, Recall@0.4 IoU of 46.5% and Mean Average Precision of 76.3% on validation data
    * Achieved total loss of 0.2199 (0.2164 objectness loss + 0.0036 regression loss) on new unseen test data
    * Achieved Precision of 90.8%, Recall @0.4IoU of 45.2, Mean Average Precision of 74.6% on new unseen test data
3. Real-time inference
    * Real-time Inference Speed using CPU (R5 7600x): 2-3 FPS 
    * Real-time Inference Speed using GPU (RTX 3060-TI): 16 FPS

# Dataset
The dataset used can be found <a href="https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset">here</a>
## Data Splitting
* 13.4k images were split into 85% for training and 15% for validation
* 3.3k additional image were used for testing

# Future Work

1. **Model Optimization**: Implement quantization and pruning for faster inference on edge devices.
2. **Multi-class Detection**: Extend the model to detect multiple object classes beyond faces.
3. **Real-time Video Processing**: Optimize the pipeline for streaming video applications.
4. **Deployment**: Create a web API and mobile application for the face detection model.




