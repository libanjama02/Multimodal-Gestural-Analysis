def data_collection(state, cap, mp_drawing, mp_hands, hand_landmarks_data, out):
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        frame_id = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # IMU data collection (from imu_recording.py)
            if state.data_quat and state.data_acc:
                count = 0
                count += 1
                # ... existing code ...

            # Hand pose data collection (from handpose_recording.py)
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                # ... existing code ...

                current_time = datetime.utcnow().isoformat()
                hand_landmarks_data.append([frame_id, current_time] + hand_landmarks)

                font = cv2.FONT_HERSHEY_SIMPLEX
                # Flip the image to write the mirrored timestamp text
                flipped_image = cv2.flip(image, 1)
                cv2.putText(flipped_image, current_time, (10,50), font, 0.7, (255, 0, 0), 1, cv2.LINE_AA)
                image = cv2.flip(flipped_image, 1)

            out.write(cv2.flip(image, 1))

            # ... existing code ...
