import cv2
import numpy as np
import face_recognition
import os
import time
import onnxruntime as ort

session = ort.InferenceSession("models/mobilenetv2-7.onnx")

def get_video_source():
    while True:
        choice = input("Would you like to access a test video (A) or your default camera (B)? (A/B): ") \
                 .upper().strip()
        if choice not in ['A', 'B']:
            print("Please enter A or B.")
            continue

        if choice == 'A':
            folder = "test_videos/"
            files = [f for f in os.listdir(folder) if f.endswith(('.mp4', '.avi', '.mov'))]

            if not files:
                print("No video files found in 'test_videos/'")
                continue

            print("\nAvailable test videos:")
            for i, file in enumerate(files):
                print(f"{i + 1}. {file}")

            while True:
                try:
                    idx = int(input("Enter the number of the video you want to use: ")) - 1
                    if 0 <= idx < len(files):
                        source = os.path.join(folder, files[idx])
                        break
                    else:
                        print("Invalid number.")
                except ValueError:
                    print("Please enter a valid number.")
            break

        else:
            source = 0  # Default webcam
            break

    return source

def start_camera(source=get_video_source()):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise Exception(f"couldn't open camera of index {source}")
    return cap

def get_frame(camera):
    ret, frame = camera.read()
    if not ret:
        raise Exception("failed to read camera frame")
    return frame

def blur_frame(frame, blur_amt=25):
    # gaussian blur kernel size must be odd
    if blur_amt % 2 == 0:
        blur_amt += 1

    return cv2.GaussianBlur(frame, (blur_amt, blur_amt), 0)

def preprocess_frame_for_MD(frame):
    # MD = motion detection
    grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(grayscaled, (21, 21), 0)

def detect_motion(prev_frame, curr_frame, min_threshold=500):
    diff = cv2.absdiff(prev_frame, curr_frame)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_areas = []
    for contour in contours:
        if cv2.contourArea(contour) < min_threshold: 
            continue
        x, y, w, h = cv2.boundingRect(contour)
        motion_areas.append((x, y, w, h))

    return motion_areas

def preprocess_face_for_MFN(img):
    # MFN = Mobile FaceNet
    """
    Preprocesses the image for Mobile FaceNet model because it needs a 
    specific input shape.
    """
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 127.5 - 1.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def get_embedding(img, session):
    """
    Returns the embedding of the face in the image using the Mobile FaceNet model.
    """
    preprocessed = preprocess_face_for_MFN(img)
    output = session.run(None, {"data": preprocessed})[0]
    embedding = output[0]
    return embedding / np.linalg.norm(embedding)

def import_faces(folder="faces/"):
    """
    Read in faces from folder and return their embeddings and names.
    """
    # Filenames should be of the form "John_Doe.jpg" or "Jane_Smith.png"
    known_encodings = []
    known_names = []

    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.png')):
            path = os.path.join(folder, filename)
            image = cv2.imread(path)

            if image is None:
                continue

            embedding = get_embedding(image, session)
            if embedding is not None:
                known_encodings.append(embedding)
                name = os.path.splitext(filename)[0].split('_')
                name = " ".join(name).title()  # Convert to "John Doe"
                known_names.append(name)

    return known_encodings, known_names

def recognize_faces(region, known_embeddings, known_names, offset=(0, 0), threshold=0.5):
    """
    Recognize faces using the face_recognition module with the region of 
    interest.
    """
    results = []
    rgb = np.ascontiguousarray(region[:, :, ::-1])

    scale = 1/4
    small_rgb = downscale(rgb, scale_factor=scale)

    offset_x, offset_y = offset

    face_locations = face_recognition.face_locations(small_rgb)
    face_locations = [
        (
            int(top / scale), 
            int(right / scale), 
            int(bottom / scale), 
            int(left / scale)
        )
        for top, right, bottom, left in face_locations
    ]

    for top, right, bottom, left in face_locations:
        face_img = rgb[top:bottom, left:right]
        if face_img.size == 0:
            continue

        embedding = get_embedding(face_img, session)
        if embedding is None:
            continue

        similarities = [np.dot(embedding, known) for known in known_embeddings]
        best_match = np.argmax(similarities)
        name = "Unknown"

        if similarities[best_match] > threshold: 
            name = known_names[best_match]

        results.append({
            "name": name, 
            "loc": (
                left + offset_x, 
                top + offset_y, 
                right + offset_x, 
                bottom + offset_y
            )
        })

    return results

def downscale(img, scale_factor=0.25):
    return cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)

def process_motion_regions(camera, frame, motion_areas, known_encodings, known_names):
    """
    Extracts regions from motion areas, expands them, and performs face recognition.
    Returns a list of tuples of recognized faces and coords (name, (x, y, w, h))
    """
    recognitions = []

    for area in motion_areas:
        x, y, w, h = area

        x_expanded, y_expanded, w_expanded, h_expanded = expand_bounding_box(x, y, w, h, frame.shape)
        
        region = frame[y_expanded:y_expanded+h_expanded, x_expanded:x_expanded+w_expanded]
        if region.size == 0:
            continue

        # Uncomment to save random debug images
        # os.makedirs("debug", exist_ok=True)
        # cv2.imwrite("debug/debug_file.jpg", region) if np.random.rand() < 0.1 else None

        recognized_faces = recognize_faces(region, known_encodings, known_names, offset=(x_expanded, y_expanded))

        for result in recognized_faces:
            name = result["name"]
            left, top, right, bottom = result["loc"]
            recognitions.append({
                "name": name,
                "loc": (left, top, right, bottom)
            })
    return recognitions

def expand_bounding_box(x, y, w, h, frame_shape, scale_x=0.2, scale_y=0.2):
    """
    Expands a box so that it's scaled relative to the full frame size,
    not the box itself.
    
    scale_x and scale_y are fractions of frame width/height to pad around the box.
    """
    frame_h, frame_w = frame_shape[:2]

    pad_w = int(frame_w * scale_x)
    pad_h = int(frame_h * scale_y)

    new_x = max(0, x - pad_w)
    new_y = max(0, y - pad_h)
    new_w = min(frame_w - new_x, w + 2 * pad_w)
    new_h = min(frame_h - new_y, h + 2 * pad_h)

    return new_x, new_y, new_w, new_h

def create_and_apply_unblur_mask(frame, blurred_frame, active_recognitions):
    """
    Creates a mask for unblurring recognized faces and applies it to blend
    clear and blurred frames together.
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for result in active_recognitions:
        left, top, right, bottom = result["loc"]
        cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)
    
    if np.any(mask):
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask_normalized = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        mask_3channel = np.stack([mask_normalized] * 3, axis=-1)
        
        return (blurred_frame * (1 - mask_3channel) + frame * mask_3channel).astype(np.uint8)
    
    return blurred_frame.copy()

def draw_recognition_overlays(frame, active_recognitions):
    display_frame = frame.copy()
    for result in active_recognitions:
        name = result["name"]
        left, top, right, bottom = result["loc"]

        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(display_frame, (left, top - 20), (right, top), (0, 255, 0), cv2.FILLED)
        cv2.putText(
            display_frame, name, (left + 5, top - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
        )

    return display_frame

def update_recognition_history(recognitions, history, max_age=1.5):
    """
    Updates recognition history with new recognitions and removes old entries.
    Returns active recognitions.
    """
    current_time = time.time()

    for rec in recognitions:
        name = rec["name"]
        box = rec["loc"]
        history[name].append((current_time, box))

    active_recognitions = []
    for name, entries in list(history.items()):
        valid_entries = [entry for entry in entries if current_time - entry[0] < max_age]
        
        if valid_entries:
            history[name] = valid_entries
            latest_box = valid_entries[-1][1]
            active_recognitions.append({
                "name": name,
                "loc": latest_box
            })
        else:
            del history[name]

    return active_recognitions

