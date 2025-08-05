import torch
import numpy as np
from torchvision import transforms, models
from torchvision.ops import nms
import cv2
import subprocess
import tempfile
import time
import json
import shutil
import sys
import os
from config import images_val_dir, labels_2_dir, artifacts_dir
import config
from preprocessing import apply_deltas_to_boxes, clamp_boxes_to_img_boundary
from models import RPN, RoIHead
from utils import generate_proposals, roi_head_inference
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import deque

class AsyncFrameProcessor:
    """
    Asynchronous frame processor for real-time object detection.
    
    This class provides an asynchronous pipeline for processing video frames through
    an object detection model (backbone + RPN + ROI head). It uses a separate thread
    to process frames in batches, enabling non-blocking frame processing suitable for
    real-time applications. The class manages input and output queues, batch processing,
    and performance tracking.
    
    Attributes:
        backbone (nn.Module): Backbone neural network for feature extraction.
        rpn_model (nn.Module): Region Proposal Network for generating object proposals.
        roi_head (nn.Module): ROI head for refining proposals and classifying objects.
        device (torch.device): Device to run the models on (CPU or GPU).
        batch_size (int): Number of frames to process in a single batch.
        score_thresh (float): Confidence threshold for filtering detections.
        nms_thresh (float): IoU threshold for Non-Maximum Suppression.
        target_fps (int): Target frames per second for processing.
        target_frame_time (float): Target processing time per frame (1/target_fps).
        input_queue (queue.Queue): Queue for input frames.
        output_queue (queue.Queue): Queue for processed results.
        running (bool): Flag indicating if the processor is running.
        processing_thread (threading.Thread): Thread for frame processing.
        frame_counter (int): Counter for total frames added.
        processed_counter (int): Counter for total frames processed.
        processing_times (deque): Tracking of recent processing times.
        last_processing_time (float): Processing time for the last batch.
    """
    def __init__(self, backbone, rpn_model, roi_head, device,
                 batch_size=4, max_queue_size=32, score_thresh=0.5, nms_thresh=0.1,
                 target_fps=30):
                 
        self.backbone = backbone
        self.rpn_model = rpn_model
        self.roi_head = roi_head
        self.device = device
        self.batch_size = batch_size
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps if target_fps > 0 else 0

        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)

        self.running = False
        self.processing_thread = None

        self.frame_counter = 0
        self.processed_counter = 0
        
        # Performance tracking
        self.processing_times = deque(maxlen=20)  # Track last 20 processing times
        self.last_processing_time = 0
    
    def start(self):
        """
        Start the frame processing thread.
        
        Creates and starts a daemon thread for processing frames. If the processor
        is already running, this method does nothing.
        """
        if not self.running:
            self.running = True
            self.processing_thread = threading.Thread(target=self._process_frames)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            print("Frame processor started")
    
    def stop(self):
        """
        Stop the frame processing thread.
        
        Sets the running flag to False and waits for the processing thread to finish
        with a timeout of 5 seconds. If the processor is not running, this method
        does nothing.
        """
        if self.running:
            self.running = False
            if self.processing_thread:
                self.processing_thread.join(timeout=5)
            print("Frame processor stopped")
    
    def add_frame(self, frame, frame_id):
        """
        Add a frame to the input queue for processing.
        
        Args:
            frame (numpy.ndarray): Input frame as a BGR image (OpenCV format).
            frame_id (int): Unique identifier for the frame.
        
        Returns:
            bool: True if the frame was successfully added to the queue, False if the queue is full.
        """
        try:
            # Use blocking put with timeout to wait if queue is full
            self.input_queue.put((frame, frame_id), block=True, timeout=0.1)
            return True
        except queue.Full:
            print("Warning: Input queue is full, dropping frame")
            return False
        
    def get_result(self):
        """
        Get a processed result from the output queue.
        
        Returns:
            tuple or None: A tuple of (frame_id, boxes, scores) if a result is available,
                        where boxes are the detected bounding boxes and scores are the
                        corresponding confidence scores. Returns None if the queue is empty.
        """
        try:
            return self.output_queue.get(block=False)
        except queue.Empty:
            return None
    
    def _process_batch(self, frames, orig_dims):
        """
        Process a batch of frames through the object detection pipeline.
        
        Args:
            frames (list): List of frames to process.
            orig_dims (list): List of original dimensions (height, width) for each frame.
        
        Returns:
            list: List of tuples (pred_boxes, pred_scores) for each frame, where pred_boxes
                are the detected bounding boxes and pred_scores are the corresponding
                confidence scores.
        
        Note:
            This method performs the following steps:
            1. Preprocesses each frame (resize, normalize, convert to tensor)
            2. Runs the frames through the backbone and RPN to generate proposals
            3. Refines proposals using the ROI head
            4. Scales the bounding boxes back to original image dimensions
            5. Tracks processing time for performance monitoring
        """
        start_time = time.time()
        
        batch_tensors = []

        for frame in frames:
            orig_h, orig_w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (512, 512))
            frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
            frame_tensor = frame_tensor.permute(2, 0, 1)
            frame_tensor = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )(frame_tensor)
            batch_tensors.append(frame_tensor)

        input_batch = torch.stack(batch_tensors).to(self.device)

        batch_size, _, H, W = input_batch.shape
        image_shapes = [(H, W)] * batch_size

        proposals_list, features = generate_proposals(
            input_batch, image_shapes, self.backbone, self.rpn_model
        )

        results = []
        for i, (proposals, (orig_h, orig_w)) in enumerate(zip(proposals_list, orig_dims)):
            
            pred_boxes, pred_scores = roi_head_inference(
                features, proposals, (H,W),
                self.roi_head, score_thresh=self.score_thresh,
                nms_thresh=self.nms_thresh
            )
            
            scale_x = orig_w / W
            scale_y = orig_h / H

            pred_boxes[:, 0] *= scale_x
            pred_boxes[:, 1] *= scale_y
            pred_boxes[:, 2] *= scale_x
            pred_boxes[:, 3] *= scale_y

            results.append((pred_boxes, pred_scores))

        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.last_processing_time = processing_time
        
        return results
    
    def _process_frames(self):
        """
        Main processing loop for the frame processing thread.
        
        Continuously collects frames from the input queue, processes them in batches,
        and places results in the output queue. The loop runs while the running flag
        is True or there are still frames in the input queue.
        
        Note:
            This method is intended to be run in a separate thread and should not be
            called directly. Use the start() method to begin processing.
        """
        batch_frames = []
        batch_frame_ids = []
        batch_orig_dims = []

        while self.running or not self.input_queue.empty():
            # Check if we should stop more frequently
            if not self.running and self.input_queue.empty():
                break
                
            while len(batch_frames) < self.batch_size and not self.input_queue.empty():
                try:
                    frame, frame_id = self.input_queue.get(block=False)
                    batch_frames.append(frame)
                    batch_frame_ids.append(frame_id)
                    batch_orig_dims.append(frame.shape[:2])
                
                except queue.Empty:
                    break
            
            if not batch_frames:
                time.sleep(0.01)
                continue

            try:
                results = self._process_batch(batch_frames, batch_orig_dims)

                for frame_id, (boxes, scores) in zip(batch_frame_ids, results):
                    self.output_queue.put((frame_id, boxes, scores))
                
                self.processed_counter += len(batch_frames)
            except Exception as e:
                print(f"Error processing batch: {e}")

            batch_frames = []
            batch_frame_ids = []
            batch_orig_dims = []
    
    def get_avg_processing_time(self):
        """
        Get the average processing time for the last few batches.
        
        Returns:
            float: Average processing time in seconds for the last 20 batches,
                or 0 if no batches have been processed yet.
        """
        if not self.processing_times:
            return 0
        return sum(self.processing_times) / len(self.processing_times)


def process_video_async(
    video_path, backbone, rpn_model,
    roi_head, device, output_path=None,
    score_thresh=0.5, nms_thresh=0.1,
    sample_every=1, batch_size=4, target_fps=30):
    """
    Process a video file asynchronously using an object detection model.
    
    This function reads a video file, processes frames through an object detection pipeline
    (backbone + RPN + ROI head) in an asynchronous manner, and displays the results with
    bounding boxes and confidence scores. The function uses the AsyncFrameProcessor class
    to handle frame processing in a separate thread, enabling real-time performance.
    
    Args:
        video_path (str): Path to the input video file.
        backbone (nn.Module): Backbone neural network for feature extraction.
        rpn_model (nn.Module): Region Proposal Network for generating object proposals.
        roi_head (nn.Module): ROI head for refining proposals and classifying objects.
        device (torch.device): Device to run the models on (CPU or GPU).
        output_path (str, optional): Path to save the processed video. Currently not implemented.
        score_thresh (float, optional): Confidence threshold for filtering detections. Default is 0.5.
        nms_thresh (float, optional): IoU threshold for Non-Maximum Suppression. Default is 0.1.
        sample_every (int, optional): Process every nth frame. Default is 1 (process all frames).
        batch_size (int, optional): Number of frames to process in a single batch. Default is 4.
        target_fps (int, optional): Target frames per second for processing. Default is 30.
    
    Returns:
        None: The function displays the processed video in a window and prints performance statistics.
    
    Note:
        The function uses FFmpeg to read the video file and OpenCV to display the results.
        It dynamically adjusts the batch size based on processing time to maintain the target FPS.
        Press 'q' to quit the video display.
    """
    temp_dir = tempfile.mkdtemp()
    ffmpeg_process = None

    try:
        fps, width, height = get_video_info(video_path)
        if fps is None:
            print("Couldn't get video info, using defaults")
            fps = 30.0
            if width is None or height is None:
                raise ValueError("Couldn't get video dimensions from ffprobe")
        
        print(f"Video info: FPS={fps}, Width={width}, Height={height}")
        print(f"Using batch size: {batch_size}, Target FPS: {target_fps}")

        ffmpeg_cmd = [
            "ffmpeg",
            "-i", video_path,
            "-f", "image2pipe",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo",
            "-"
        ]

        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE, bufsize=10**8)
        backbone.eval()
        rpn_model.eval()
        roi_head.eval()

        frame_count = 0
        processed_count = 0
        displayed_count = 0
        
        frame_size = width * height * 3

        processor = AsyncFrameProcessor(
            backbone, rpn_model, roi_head, device,
            batch_size=batch_size, score_thresh=score_thresh, nms_thresh=nms_thresh,
            target_fps=target_fps
        )
        processor.start()

        frames_in_flight = {}
        next_frame_id = 0
        next_display_id = 0

        start_time = time.time()
        last_display_time = time.time()
        last_frame_read_time = time.time()
        print("Processing video asynchronously ... press 'q' to quit")

        # Flow control variables
        max_frames_to_buffer = 8  # Reduced buffer size for more stable FPS
        video_ended = False
        
        # FPS stabilization variables
        frame_display_times = deque(maxlen=30)  # Track last 30 frame display times
        dynamic_batch_size = batch_size
        min_batch_size = 1
        max_batch_size = 8

        while True:
            loop_start_time = time.time()
            
            # Only read new frames if we haven't buffered too many ahead
            if not video_ended and (next_frame_id - next_display_id) < max_frames_to_buffer:
                raw_frame_data = ffmpeg_process.stdout.read(frame_size)
                
                if len(raw_frame_data) != frame_size:
                    if ffmpeg_process.poll() is not None and len(raw_frame_data) == 0:
                        print("FFmpeg process finished, end of video stream")
                        video_ended = True
                    else:
                        print("No more data from ffmpeg stdout, breaking loop")
                        video_ended = True
                else:
                    frame = np.frombuffer(raw_frame_data, dtype=np.uint8).reshape((height, width, 3)).copy()

                    if frame_count % sample_every == 0:
                        frames_in_flight[next_frame_id] = frame.copy()
                        # Use blocking put with timeout to wait if queue is full
                        if processor.add_frame(frame, next_frame_id):
                            next_frame_id += 1
                        else:
                            # If we couldn't add the frame, remove it from in-flight
                            del frames_in_flight[next_frame_id]
                    
                    frame_count += 1
                    last_frame_read_time = time.time()

            # Process results and display
            result = processor.get_result()
            if result is not None:
                frame_id, boxes, scores = result

                if frame_id == next_display_id:
                    if frame_id in frames_in_flight:
                        display_frame = frames_in_flight[frame_id]

                        for box, score in zip(boxes, scores):
                            x1, y1, x2, y2 = box.int().tolist()
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, f"{score:.2f}", (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        current_time = time.time()
                        
                        # Calculate and display FPS
                        frame_display_times.append(current_time)
                        if len(frame_display_times) >= 2:
                            # Calculate FPS based on the time it took to display the last few frames
                            time_span = frame_display_times[-1] - frame_display_times[-min(10, len(frame_display_times))]
                            frames_in_span = min(10, len(frame_display_times) - 1)
                            if time_span > 0:
                                fps_display = frames_in_span / time_span
                            else:
                                fps_display = 0
                            
                            cv2.putText(display_frame, f"FPS: {fps_display:.1f}", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Display processing stats
                        avg_proc_time = processor.get_avg_processing_time()
                        cv2.putText(display_frame, f"Avg Proc: {avg_proc_time*1000:.1f}ms", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        last_display_time = current_time
                        cv2.imshow('Face Detection', display_frame)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("'q' pressed, quitting...")
                            break

                        displayed_count += 1

                        del frames_in_flight[frame_id]
                        next_display_id += 1

                        if displayed_count % 10 == 0:
                            elapsed_total = current_time - start_time
                            avg_fps_total = displayed_count / elapsed_total if elapsed_total > 0 else 0
                            print(f"Displayed {displayed_count} frames, Average FPS: {avg_fps_total:.1f}")
                    
                    else:
                        print(f"Warning: Frame {frame_id} not found in in-flight")
                        next_display_id += 1
                
                elif frame_id < next_display_id:
                    if frame_id in frames_in_flight:
                        del frames_in_flight[frame_id]
                else:
                    # Put the result back in the queue if it's a future frame
                    processor.output_queue.put(result)
            else:
                # No result available, sleep briefly to avoid busy waiting
                time.sleep(0.005)

            # Check if we should exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed, quitting...")
                break
            
            # If video has ended and we've processed all frames, exit
            if video_ended and next_display_id >= next_frame_id:
                print("All frames processed")
                break
            
            # Dynamic batch size adjustment based on processing time
            if displayed_count > 0 and displayed_count % 20 == 0:
                avg_proc_time = processor.get_avg_processing_time()
                if avg_proc_time > 0:
                    # Calculate ideal batch size to maintain target FPS
                    ideal_batch_time = 1.0 / target_fps if target_fps > 0 else avg_proc_time
                    ideal_batch_size = max(min_batch_size, min(max_batch_size, 
                                                              int(ideal_batch_time / avg_proc_time * dynamic_batch_size)))
                    
                    if ideal_batch_size != dynamic_batch_size:
                        print(f"Adjusting batch size from {dynamic_batch_size} to {ideal_batch_size}")
                        dynamic_batch_size = ideal_batch_size
                        # Update processor batch size
                        processor.batch_size = dynamic_batch_size

        print("Waiting for remaining frames to be processed...")
        timeout = 10  # Increased timeout
        start_wait = time.time()

        while next_display_id < next_frame_id and (time.time() - start_wait) < timeout:
            result = processor.get_result()
            if result is None:
                time.sleep(0.01)
                continue
            
            frame_id, boxes, scores = result
            if frame_id == next_display_id and frame_id in frames_in_flight:
                display_frame = frames_in_flight[frame_id]

                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = box.int().tolist()
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{score:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                cv2.imshow('Face Detection', display_frame)
                cv2.waitKey(1)

                displayed_count += 1
                del frames_in_flight[frame_id]
                next_display_id += 1
            
            elif frame_id < next_display_id and frame_id in frames_in_flight:
                del frames_in_flight[frame_id]
            else:
                processor.output_queue.put(result)
        
        total_time = time.time() - start_time
        avg_fps = displayed_count / total_time if total_time > 0 else 0
        print(f"Processing complete. Displayed {displayed_count} frames in {total_time:.1f} seconds")
        print(f"Overall Average FPS: {avg_fps:.1f}")
    
    finally:
        # Stop the processor first
        if 'processor' in locals():
            processor.stop()
        
        # Then terminate ffmpeg
        if ffmpeg_process and ffmpeg_process.poll() is None:
            ffmpeg_process.terminate()
            try:
                ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ffmpeg_process.kill()
        
        # Clean up
        cv2.destroyAllWindows()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def get_video_info(video_path):
    """
    Extract metadata from a video file using FFprobe.
    
    This function uses FFprobe to retrieve video metadata including frame rate,
    width, and height. It handles potential errors in parsing the metadata and
    provides default values when necessary.
    
    Args:
        video_path (str): Path to the video file.
    
    Returns:
        tuple: A tuple containing (fps, width, height) where:
            - fps (float): Frames per second of the video.
            - width (int): Width of the video in pixels.
            - height (int): Height of the video in pixels.
            If the information cannot be retrieved, returns (None, None, None).
    
    Note:
        The function uses FFprobe in JSON output format for reliable parsing.
        It handles potential errors in parsing the frame rate by using eval() with
        a fallback to 30.0 fps if parsing fails.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        info = json.loads(result.stdout)
        video_stream = next((stream for stream in info['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream:
            try:
                fps = float(eval(video_stream.get('r_frame_rate', '30/1')))
            except (SyntaxError, NameError):
                fps = 30.0

            width = video_stream['width']
            height = video_stream['height']
            return fps, width, height
        return None, None, None
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, StopIteration) as e:
        print(f"Error getting video info: {e}") 
        return None, None, None

if __name__ == "__main__":
    
    models_dir = os.path.join(os.getcwd(), "artifacts/")
    vids_dir = os.path.join(os.getcwd(), "test_vids/")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rpn_model = RPN()
    rpn_model.load_state_dict(torch.load(models_dir + "rpn_5epchs_vgg16bb_s2_r1_lr1e-4_wghts_bs16.pth"))
    backbone = torch.load(models_dir + "vgg16_backbone.pth", weights_only=False)
    roi_head = RoIHead(in_channels=512, num_classes=1)
    roi_head.load_state_dict(torch.load(models_dir + "roi_10eps_rpns2r1_2048fc_512p_07fg_015val_wghts.pth"))

    for p in rpn_model.parameters():
        p.requires_grad = False
    for p in backbone.parameters():
        p.requires_grad = False
    for p in roi_head.parameters():
        p.requires_grad = False

    # Set target FPS based on your requirements
    target_fps = 16  # Adjust based on your system capabilities
    
    process_video_async(
        video_path=vids_dir + "trailer.mp4",
        backbone=backbone.to(device),
        rpn_model=rpn_model.to(device),
        roi_head=roi_head.to(device),
        device=device,
        score_thresh=0.5,
        nms_thresh=0.1,
        sample_every=1,
        batch_size=4,  # Initial batch size
        target_fps=target_fps
    )