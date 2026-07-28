import torch
import statistics
from collections import defaultdict
from torchvision.ops import box_iou

def smooth_classes_by_mode(class_history_list, fallback_class):
    """
    Takes a list of raw class predictions for a specific track ID and 
    returns the most frequently occurring class (the mode).

    Parameters
    ----------
    class_history_list : list of int
        List of historical class IDs assigned to a specific track.
    fallback_class : int
        The class ID to return if the history list is empty (e.g., the root class).

    Returns
    -------
    int
        The most frequent class ID (mode). If there is a tie, falls back to the 
        most recent class prediction.
    """
    if not class_history_list:
        return fallback_class
        
    try:
        return statistics.mode(class_history_list)
    except statistics.StatisticsError:
        return class_history_list[-1]

def class_agnostic_track(model, source_path, inference_args, iou_threshold=0.85):
    """
    A generator that wraps model.track(). 
    
    It applies the class-agnostic callback, runs the tracker, and then mathematically 
    matches the tracked boxes back to the raw detections using IoU to recover the 
    true classes and soft_scores.

    Parameters
    ----------
    model : ultralytics.engine.model.Model
        The loaded Ultralytics YOLO model to be used for tracking.
    source_path : str
        The direct path to the input images or video source.
    inference_args : dict
    iou_threshold : float
        The minimum IoU overlap required to match a tracked box to a raw detection.
    
    yields
    ------
    ultralytics.engine.results.Results
        Ultralytics Result objects, but with classes and soft_scores recovered.
        
    Raises
    ------
    ValueError
        If the model does not have a valid hierarchy with a root node.
    """
    
    # 1. Auto-discover the root class to use as the fallback for ghost tracks
    hierarchy = getattr(model, 'hierarchy', None)
    if hierarchy is None and hasattr(model, 'model'):
        hierarchy = getattr(model.model, 'hierarchy', None)
        
    if hierarchy is None or not hasattr(hierarchy, 'roots') or len(hierarchy.roots) == 0:
        raise ValueError("Class-agnostic tracking requires a valid hierarchy with a defined root class.")
        
    dummy_class = int(hierarchy.roots[0].item())
    
    # This list acts as a synchronized, thread-safe FIFO queue between the callback and the generator loop.
    # It completely bypasses the tracker's destructive slicing of the `result` object.
    frame_stash = []
        
    def trick_tracker_callback(predictor):
        """
        Closure callback that pushes pristine detections onto the local queue 
        before the tracker has a chance to mangle them.
        """
        for result in predictor.results:
            # Prepare an empty stash dictionary for this specific frame
            stash = {
                "orig_boxes": None,
                "orig_classes": None,
                "orig_soft_scores": None
            }
            
            if result.boxes is not None and len(result.boxes) > 0:
                # Stash the unmodified coordinates and classes inside our local dictionary
                stash["orig_boxes"] = result.boxes.xyxy.clone()
                stash["orig_classes"] = result.boxes.cls.clone()
                
                # Explicitly stash soft_scores if they exist
                if hasattr(result, 'soft_scores') and result.soft_scores is not None:
                    stash["orig_soft_scores"] = result.soft_scores.clone()
                
                # Force all classes to the dynamically discovered dummy class
                # Clone the data tensor to avoid PyTorch InferenceMode inplace update errors
                new_data = result.boxes.data.clone()
                new_data[:, -1] = dummy_class
                result.boxes.data = new_data
                
            # Push the frame's pristine data onto the queue
            frame_stash.append(stash)

    # 2. Attach our closure callback
    model.clear_callback("on_predict_postprocess_end")
    model.add_callback("on_predict_postprocess_end", trick_tracker_callback)
    
    # Track ID -> List of raw integer classes mapped to that ID
    track_class_history = defaultdict(list)
    
    # 3. Execute the tracking stream
    results_stream = model.track(**inference_args)
    
    for result in results_stream:
        # Pop the oldest stash off the queue (FIFO)
        # We MUST pop even if the frame is empty to keep the queue perfectly synchronized!
        current_stash = frame_stash.pop(0) if frame_stash else {}
        
        # If no objects were tracked in this frame, just yield the empty result
        if result.boxes is None or result.boxes.id is None:
            yield result
            continue
            
        tracked_boxes = result.boxes.xyxy
        tracked_ids = result.boxes.id.int().cpu().tolist()
        
        # Retrieve the original detections securely from our local queue instead of the result object
        orig_boxes = current_stash.get("orig_boxes")
        orig_classes = current_stash.get("orig_classes")
        orig_soft_scores = current_stash.get("orig_soft_scores")
        
        # We must re-align classes/scores even if orig_boxes is empty, 
        # because the tracker might be outputting pure "ghost tracks"
        smoothed_cls_tensor = torch.zeros_like(result.boxes.cls)
        
        # Prepare empty tensor for explicit soft_scores (zeros for ghost tracks)
        new_soft_scores = None
        if orig_soft_scores is not None:
            new_shape = (len(tracked_ids), orig_soft_scores.shape[1])
            new_soft_scores = torch.zeros(new_shape, device=orig_soft_scores.device, dtype=orig_soft_scores.dtype)

        # Calculate IoU overlaps only if we have original boxes to match against
        if orig_boxes is not None and len(orig_boxes) > 0:
            ious = box_iou(tracked_boxes, orig_boxes)
        else:
            ious = torch.zeros((len(tracked_ids), 0), device=result.boxes.data.device)
            
        for i, track_id in enumerate(tracked_ids):
            # If we have original boxes, find the index of the best overlap
            if ious.shape[1] > 0:
                best_match_idx = torch.argmax(ious[i]).item()
                max_iou = ious[i][best_match_idx].item()
                
                # If the IoU is strong enough, it's a valid match (not a ghost track)
                if max_iou > iou_threshold:
                    real_class = int(orig_classes[best_match_idx].item())
                    track_class_history[track_id].append(real_class)
                    
                    # Re-align explicit soft_scores
                    if new_soft_scores is not None:
                        new_soft_scores[i] = orig_soft_scores[best_match_idx]
            
            # Retrieve from history (This inherently covers both matches and ghost tracks)
            smoothed_class = smooth_classes_by_mode(track_class_history[track_id], fallback_class=dummy_class)
            smoothed_cls_tensor[i] = smoothed_class
                
        # 4. Mutate the result object to inject the smoothed classes back in
        # Clone the data tensor to avoid PyTorch InferenceMode inplace update errors
        new_data = result.boxes.data.clone()
        new_data[:, -1] = smoothed_cls_tensor
        result.boxes.data = new_data
        
        # 5. Inject the re-aligned explicit soft_scores back into the result object
        if new_soft_scores is not None:
            result.soft_scores = new_soft_scores
            
        yield result
        
    # Clean up the callback when the stream finishes
    model.clear_callback("on_predict_postprocess_end")
