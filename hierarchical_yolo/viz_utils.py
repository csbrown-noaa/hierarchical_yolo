import numpy as np
import cv2
from PIL import Image

def rescale_boxes(pred_boxes, from_shape, to_shape):
    """
    Rescales predicted boxes from model input shape to original image shape.

    Args:
        pred_boxes (np.ndarray): (R, 4) array of [x1, y1, x2, y2]
        from_shape (tuple): (height, width) of model input, e.g., (640, 640)
        to_shape (tuple): (height, width) of original image

    Returns:
        np.ndarray: Rescaled boxes of shape (R, 4)
    """
    
    gain_w = to_shape[1] / from_shape[1]
    gain_h = to_shape[0] / from_shape[0]
    print(from_shape, to_shape)
    print(gain_w, gain_h)

    pred_boxes[..., [0,2]] *= gain_w
    pred_boxes[..., [1,3]] *= gain_h
    return pred_boxes
    

def draw_boxes_on_image(pil_img, boxes, labels=None, scores=None, box_color=(0, 255, 0), text_color=(255, 255, 255), scaled=(640,640)):
    """
    Draws bounding boxes with optional labels and scores onto an image.

    Args:
        pil_img (PIL.Image): Original image
        boxes (np.ndarray): (R, 4) array of boxes [x1, y1, x2, y2]
        labels (list of str): List of class names for each box
        scores (list of float): Confidence scores
        box_color (tuple): BGR color for boxes
        text_color (tuple): BGR color for text

    Returns:
        PIL.Image: Annotated image
    """
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    img_area = pil_img.size[::-1][0] * pil_img.size[::-1][1]
    expected_area = 1080*1920
    area_scale_factor = 4 * img_area / expected_area 


    for i, box in enumerate(boxes):
        #box = xywh2xyxy(box.clone())
        box = rescale_boxes(box.clone(), scaled, pil_img.size[::-1])

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), box_color, 2)

        label_text = ''
        if labels is not None:
            label_text += labels[i]
        if scores is not None:
            label_text += f' {scores[i]:.2f}' if label_text else f'{scores[i]:.2f}'

        if label_text:
            cv2.putText(img_cv, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, area_scale_factor, text_color, 2)

    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

