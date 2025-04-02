import scripts
import time 
import cv2
from collections import defaultdict

def main():
    print("Initializing camera...")
    camera = scripts.start_camera()
    
    print("Loading face database...")
    faces, names = scripts.import_faces()
    if not faces:
        print("No faces found in the database.")
        return
    print(f"Loaded {len(faces)} faces from the database.")

    prev_frame = None
    recognition_history = defaultdict(list)
    
    print("Starting main loop. Press 'q' to quit.")
    try:
        frame_counter = 0
        while True:
            frame = scripts.get_frame(camera)
            frame_counter += 1
            blurred = scripts.blur_frame(frame, blur_amt=80)
            processed = scripts.preprocess_frame_for_MD(blurred)

            if prev_frame is None:
                prev_frame = processed
                continue

            display_frame = blurred
            recognized_faces = []
            recognitions = []

            # every 10 frames:
            if frame_counter % 10 == 0:
                motion_areas = scripts.detect_motion(prev_frame, processed, min_threshold=500)
                prev_frame = processed

                recognitions = scripts.process_motion_regions(camera, frame, motion_areas, faces, names)
                for recognition in recognitions:
                    if recognition['name'] not in recognized_faces:
                        recognized_faces.append(recognition['name'])

            active_recognitions = scripts.update_recognition_history(recognitions, recognition_history)

            display_frame = scripts.create_and_apply_unblur_mask(frame, blurred, active_recognitions)
            display_frame = scripts.draw_recognition_overlays(display_frame, active_recognitions)

            cv2.imshow("Face Recognition", display_frame)
            if recognized_faces:
                print(f"Recognized faces at frame {frame_counter}:", recognized_faces)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user")
    # except Exception as e:
    #     print("An error occurred:", e)
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Exiting program")

if __name__ == '__main__':
    main()
    
