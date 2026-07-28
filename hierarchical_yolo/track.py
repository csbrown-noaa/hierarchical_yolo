import torch
from torchvision.ops import box_iou

def class_agnostic_track(model, source_path, inference_args, iou_threshold=0.5):
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
        The minimum IoU overlap required to match a tracked box back to a raw detection.
        
    yields
    ------
    ultralytics.engine.results.Results
        Ultralytics Result objects, but with classes and soft_scores recovered.
        
    Raises
    ------
    ValueError
        If the model does not have a valid hierarchy with a root node.
    """
    hierarchy = getattr(model, 'hierarchy', None)
    if hierarchy is None and hasattr(model, 'model'):
        hierarchy = getattr(model.model, 'hierarchy', None)
        
    if hierarchy is None or not hasattr(hierarchy, 'roots') or len(hierarchy.roots) == 0:
        raise ValueError("Class-agnostic tracking requires a valid hierarchy with a defined root class.")
        
    dummy_class = int(hierarchy.roots[0].item())
        
    # 2. Setup the thread-safe local stash (closure pattern)
    frame_stash = []
    
    def trick_tracker_callback(predictor):
        """
        Callback executed right after NMS. Stashes original data into the local 
        generator scope to survive the tracker's destructive tensor slicing.
        """
        for result in predictor.results:
            if result.boxes is not None and len(result.boxes) > 0:
                # Stash the unmodified coordinates, classes, and confidence scores
                stash = {
                    "orig_boxes": result.boxes.xyxy.clone(),
                    "orig_classes": result.boxes.cls.clone(),
                    "orig_conf": result.boxes.conf.clone(),
                    "orig_soft_scores": None
                }
                
                # Explicitly stash soft_scores if they exist
                if hasattr(result, 'soft_scores') and result.soft_scores is not None:
                    stash["orig_soft_scores"] = result.soft_scores.clone()
                    
                # Clone the data tensor to bypass PyTorch InferenceMode restrictions
                result.boxes.data = result.boxes.data.clone()
                # Force all classes to the dynamically discovered dummy class
                result.boxes.data[:, -1] = dummy_class
                
                frame_stash.append(stash)
            else:
                frame_stash.append({})

    # Safely clear the previous callback to prevent memory leaks or recursion
    model.clear_callback("on_predict_postprocess_end")
    model.add_callback("on_predict_postprocess_end", trick_tracker_callback)
    
    # 3. Execute the tracking stream
    results_stream = model.track(**inference_args)
    
    for result in results_stream:
        # Pop the synchronized original data for this exact frame
        current_stash = frame_stash.pop(0) if frame_stash else {}
        
        # If no objects were tracked in this frame, just yield the empty result
        if result.boxes is None or result.boxes.id is None:
            yield result
            continue
            
        tracked_boxes = result.boxes.xyxy
        tracked_conf = result.boxes.conf
        tracked_ids = result.boxes.id.int().cpu().tolist()
        
        # Retrieve the stashed original detections
        orig_boxes = current_stash.get("orig_boxes")
        orig_classes = current_stash.get("orig_classes")
        orig_conf = current_stash.get("orig_conf")
        orig_soft_scores = current_stash.get("orig_soft_scores")
        
        # Default all tracks to dummy_class. Only matched tracks will be overwritten with true classes.
        smoothed_cls_tensor = torch.full_like(result.boxes.cls, dummy_class)
        
        # Prepare explicit soft_scores initialized with NaNs for unmatched ghost tracks
        new_soft_scores = None
        if orig_soft_scores is not None:
            new_shape = (len(tracked_ids), orig_soft_scores.shape[1])
            new_soft_scores = torch.full(new_shape, float('nan'), device=orig_soft_scores.device, dtype=orig_soft_scores.dtype)

        # Calculate spatial IoU overlaps
        if orig_boxes is not None and len(orig_boxes) > 0:
            ious = box_iou(tracked_boxes, orig_boxes)
        else:
            ious = torch.zeros((len(tracked_ids), 0), device=result.boxes.data.device)
            
        for i, track_id in enumerate(tracked_ids):
            # If we have original boxes, find the index of the best overlap
            if ious.shape[1] > 0:
                # Create a mask for candidates meeting BOTH spatial and confidence criteria
                # Using 1e-4 tolerance for floating point comparisons of confidence scores
                valid_mask = (ious[i] > iou_threshold) & (torch.abs(orig_conf - tracked_conf[i]) < 1e-4)
                
                if valid_mask.any():
                    # Filter IoUs to only consider valid candidates, then find highest spatial overlap
                    valid_ious = ious[i].clone()
                    valid_ious[~valid_mask] = 0.0
                    best_match_idx = torch.argmax(valid_ious).item()
                    
                    real_class = int(orig_classes[best_match_idx].item())
                    smoothed_cls_tensor[i] = real_class
                    
                    # Re-align explicit soft_scores
                    if new_soft_scores is not None:
                        new_soft_scores[i] = orig_soft_scores[best_match_idx]
                
        # Clone again to bypass PyTorch InferenceMode restrictions before mutation
        result.boxes.data = result.boxes.data.clone()
        result.boxes.data[:, -1] = smoothed_cls_tensor
        
        # Inject the re-aligned explicit soft_scores back into the result object
        if new_soft_scores is not None:
            result.soft_scores = new_soft_scores
            
        yield result
        
    # Clean up the callback when the stream finishes
    model.clear_callback("on_predict_postprocess_end")
