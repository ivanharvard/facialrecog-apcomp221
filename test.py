import cv2
import scripts

def main():
    source = scripts.get_video_source()
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("couldn't open camera")
        return
    
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't read frame. Video or camera feed may have ended.")
            break
        
        # blur with Gaussian blur kernel
        blurred_frame = cv2.GaussianBlur(frame, (25,25), 0)

        cv2.imshow('camera feed', frame)
        cv2.imshow('blurred frame', blurred_frame)

        # PRESS Q TO EXIT
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


main()