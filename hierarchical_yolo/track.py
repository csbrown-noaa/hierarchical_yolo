import torch
import statistics
from collections import defaultdict
from torchvision.ops import box_iou

def trick_tracker_callback(predictor):
    """
    Callback function to be attached to the Ultralytics YOLO model.
    
    It executes right after NMS, before the tracker processes the detections.
    It stashes the original boxes and classes, then forces all classes to 0 
    so the tracker becomes completely class-agnostic.

    Parameters
    ----------
    predictor : ultralytics.engine.predictor.BasePredictor
        The YOLO predictor instance containing the current batch results.
    """
    for result in predictor.results:
        if result.boxes is not None and len(result.boxes) > 0:
            # 1. Stash the unmodified coordinates and classes inside the result object
            result.orig_boxes_copy = result.boxes.xyxy.clone()
            result.orig_classes_copy = result.boxes.cls.clone()
            
            # 2. Force all classes to 0. This tricks ByteTrack/BoT-SORT into 
            # treating every object as the same category, preventing lost tracks 
            # when the raw predictions flicker between categories.
            result.boxes.cls = torch.zeros_like(result.boxes.cls)

def smooth_classes_by_mode(class_history_list):
    """
    Takes a list of raw class predictions for a specific track ID and 
    returns the most frequently occurring class (the mode).

    Parameters
    ----------
    class_history_list : list of int
        List of historical class IDs assigned to a specific track.

    Returns
    -------
    int
        The most frequent class ID (mode). If there is a tie, falls back to the 
        most recent class prediction.
    """
    if not class_history_list:
        return 0
    try:
        return statistics.mode(class_history_list)
    except statistics.StatisticsError:
        # If there is a tie for the mode, fallback to the most recent class prediction
        return class_history_list[-1]

def class_agnostic_track(model, source_path, inference_args, iou_threshold=0.5):
    """
    A generator that wraps model.track(). 
    
    It applies the class-agnostic callback, runs the tracker, and then mathematically 
    matches the tracked boxes back to the raw detections using IoU to recover the 
    true classes.

    Parameters
    ----------
    model : ultralytics.engine.model.Model
        The loaded Ultralytics YOLO model to be used for tracking.
    source_path : str
        The direct path to the input images or video source.
    inference_args : dict
        A dictionary of keyword arguments to pass directly to `model.track()`.
    iou_threshold : float, optional
        The minimum IoU overlap required to match a tracked box to an original 
        raw detection (default is 0.5).

    Yields
    ------
    ultralytics.engine.results.Results
        Ultralytics Result objects, but with `result.boxes.cls` updated to the 
        smoothed mode of the object's class history.
    """
    
    # 1. Attach our callback
    model.add_callback("on_predict_postprocess_end", trick_tracker_callback)
    
    # Track ID -> List of raw integer classes mapped to that ID
    track_class_history = defaultdict(list)
    
    # 2. Execute the tracking stream
    results_stream = model.track(**inference_args)
    
    for result in results_stream:
        # If no objects were tracked in this frame, just yield the empty result
        if result.boxes is None or result.boxes.id is None:
            yield result
            continue
            
        tracked_boxes = result.boxes.xyxy
        tracked_ids = result.boxes.id.int().cpu().tolist()
        
        # Retrieve the stashed original detections that survived NMS
        orig_boxes = getattr(result, "orig_boxes_copy", None)
        orig_classes = getattr(result, "orig_classes_copy", None)
        
        if orig_boxes is not None and len(orig_boxes) > 0:
            # Calculate a matrix of IoU overlaps: Shape [Num_Tracked_Boxes, Num_Original_Boxes]
            ious = box_iou(tracked_boxes, orig_boxes)
            
            # Create a tensor to hold the newly smoothed classes we are about to calculate
            smoothed_cls_tensor = torch.zeros_like(result.boxes.cls)
            
            for i, track_id in enumerate(tracked_ids):
                # Find the index of the original raw box that most overlaps this tracked box
                best_match_idx = torch.argmax(ious[i]).item()
                max_iou = ious[i][best_match_idx].item()
                
                # If the IoU is strong enough, it's a valid match (not a ghost track)
                if max_iou > iou_threshold:
                    # Grab the original, unmodified class and save it to history
                    real_class = int(orig_classes[best_match_idx].item())
                    track_class_history[track_id].append(real_class)
                
                # Calculate the smoothed class based on the entire history of this track
                smoothed_class = smooth_classes_by_mode(track_class_history[track_id])
                smoothed_cls_tensor[i] = smoothed_class
                
            # 3. Mutate the result object to inject the smoothed classes back in
            result.boxes.cls = smoothed_cls_tensor
            
        yield result
        
    # Clean up the callback when the stream finishes
    model.clear_callbacks()
