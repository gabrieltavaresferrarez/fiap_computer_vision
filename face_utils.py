import cv2
import numpy as np

def crop_face(frame, bbox, resize_factor = 1):
    x, y, w, h = bbox
    x = int(x*resize_factor)
    y = int(y*resize_factor)
    w = int(w*resize_factor)
    h = int(h*resize_factor)
    return frame[y:y+h, x:x+w]

def draw_bbox(frame, bbox, resize_factor = 1):
    x, y, w, h = bbox
    x = int(x*resize_factor)
    y = int(y*resize_factor)
    w = int(w*resize_factor)
    h = int(h*resize_factor)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame