"""
def main():
    camera = start_camera()
    while camera is going:
        frame = get_frame(camera)
        blurred_frame = cv2.GaussianBlur(frame, (25,25), 0)

        motion_areas = detect_motion(frame)

        mask = np.zeros(frame.shape[:,2]) # for unblurring
        for area in motion_areas:
            x, y, z, h = area
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)

        display_frame = frame.copy()
        display_frame[mask > 0] = frame[mask > 0]

        if motion_areas:
            people = detect_faces(frame, motion_areas, faces_database)
            display_names(people)

"""