import numpy as np
import cv2

# get coorner positions (x,y) of a square defined pby (x, y width, height)
def get_corners(x, y, w, h):
    return np.array([
        [x,     y    ],  # top-left
        [x + w, y    ],  # top-right
        [x,     y + h],  # bottom-left
        [x + w, y + h],  # bottom-right
    ], dtype=np.float32)

# get the mean bbox out of a lista of bbox
def smooth_bboxes(list_bbox, outlier_threshold=10, verbose=False):
    if verbose:
        print(f'Verbose activated in smooth_bboxes function. Args: outlier_threshold : {outlier_threshold}')
    if not list_bbox:
        return None

    list_frameCorners = [get_corners(x, y, w, h) for (x, y, w, h) in list_bbox]
    if verbose:
        print('''Corners detected:''')
        for i, frame in enumerate(list_frameCorners):
            print(f'Frame {i}:')
            for corner in frame:
                print(f'\t{corner}')

    # mean_corners = np.mean(list_frameCorners, axis=0) # --> sensible to noise
    median_corners = np.median(list_frameCorners, axis=0)
    if verbose:
        print('Mean of corners:')
        for corner in median_corners:
            print(f'\t{corner}')

    filtered = []
    for corners in list_frameCorners:
        distances = np.linalg.norm(corners - median_corners, axis=1)
        if np.all(distances <= outlier_threshold):
            filtered.append(corners)

    if verbose:
        print(filtered)
        print(f'Frames passed filter:')
        for frame in filtered:
            print(f'Frame:')
            for corner in frame:
                print(f'\t{corner}')

    if not filtered:
        return None

    mean_filtered = np.mean(filtered, axis=0)

    if verbose:
        print('Mean of corners filtered:')
        for corner in mean_filtered:
            print(f'\t{corner}')

    x  = int(mean_filtered[0][0])
    y  = int(mean_filtered[0][1])
    w  = int(mean_filtered[1][0] - x)
    h  = int(mean_filtered[2][1] - y)
    return (x, y, w, h)


# converts a list of RGB frames into Grayscale
def to_gray_scale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# receive an image e return a list of all bbox of faces detected
def get_face_bbox(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    bbox = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
    return list(bbox)

# receive a list o bbox in one frame and select the biggets one by the height
def get_main_bbox(list_bbox):
    if list_bbox is None or len(list_bbox) == 0:
        return None
    best = max(list_bbox, key=lambda bbox: bbox[3]) # best height --> 3 dimension
    return tuple(best)  # garante tupla Python, não numpy array

# get a window of window_size frames in a list
def get_frames_window(capture, window_size=10):
    list_frames = []
    for i in range(window_size):
        ret, frame = capture.read()
        if ret:
            list_frames.append(frame)
    return list_frames

# receive a frame and rreturn the resized frame
def get_resized(frame, resize_factor):
    return cv2.resize(frame, (int(frame.shape[1] * resize_factor), int(frame.shape[0] * resize_factor)))


# receives a list of frames and returns the main box of face in this frames window
def get_face_bbox_framesWindow(list_framesWindow, resize_factor = 0.5):

    list_framesWindow = get_resized(list_framesWindow, resize_factor)
    list_framesWindowGray = to_gray_scale(list_framesWindow)

    list_bbox = []
    for frame in list_framesWindowGray:
        bbox = get_face_bbox(frame) # get all bbox detected in frame
        main_bbox = get_main_bbox(bbox)
        if main_bbox:
            list_bbox.append(main_bbox) # filter only the main bbox in frame 
    
    bbox_face = smooth_bboxes(list_bbox)
    return bbox_face



# ------------------------
# funções atualizadas!
# ------------------------







def capture_face(frame, face_path): # salva o frame no path
    cv2.imwrite(face_path, frame)
    print(f'Frame salvo em: {face_path}')
    return True






# MEDIAPIPE para visualização de piscada
import cv2
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks import python
model_path = "face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)


def get_eye_state(frame, resize_ratio=0.3): # retorna o estado do olho  OPEN, CLOSED, NOT_DETECTED
    # frame_process = to_gray_scale(frame)
    frame_process = get_resized(frame, resize_ratio)

    frame_process = cv2.cvtColor(frame_process, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_process
    )
    result = detector.detect(mp_image)

    if result.face_blendshapes:
        # índices dos olhos fechando (EAR-like via blendshapes)
            left_eye = result.face_blendshapes[0][9].score
            right_eye = result.face_blendshapes[0][10].score

            avg = (left_eye + right_eye) / 2

            if avg > 0.6:
                return 'CLOSED'
            else:# < 0.2:
                return 'OPEN'

    return 'NOT_DETECTED'  # implementar


def detect_face(frame, resize_ratio = 0.3):

    frame_process = get_resized(frame, resize_ratio)
    frame_process = cv2.cvtColor(frame_process, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_process
    )
    result = detector.detect(mp_image)

    if len(result.face_landmarks) > 0:
        return True  # implementar
    else:
        return False

### --------------------------------------------------------------------------------------------