from setup import mouse, mp_hands, mp_drawing, mp_drawing_styles, screen_width, screen_height, cap, image
import functions as fn

from pynput.mouse import Button
import cv2
import numpy as np

# Mouse click state and sensitivity
mouse_pressed = False
sensitivity_factor = 3.0
margin = 300


# Create a Hands object.
with mp_hands.Hands(
        max_num_hands=1,  # Assuming you want to detect the pinch in one hand.
        min_detection_confidence=0.3,
        min_tracking_confidence=0.5) as hands:
    while True:
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # Draw hand connections.
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())


                # Get coordinates and smooth landmarks using EMA
                coordinates = fn.get_landmark_coordinates(hand_landmarks, image.shape, mp_hands)
                smoothed_coordinates = fn.smooth_landmarks(coordinates)

                # Calculate midpoints and distances
                midpoints = fn.calculate_midpoints(coordinates)
                distances = fn.calculate_distances(coordinates)

                # Control mouse cursor with index-finger tip
                mouse_x, mouse_y = midpoints[('THUMB_TIP', 'INDEX_FINGER_TIP')]
                screen_x = np.interp(mouse_x, [margin, image.shape[1] - margin],
                                     [0, screen_width / sensitivity_factor])
                screen_y = np.interp(mouse_y, [margin, image.shape[0] - margin],
                                     [0, screen_height / sensitivity_factor])
                mouse.position = (screen_x * sensitivity_factor, screen_y * sensitivity_factor)

                # Smooth the mouse movements
                smoothed_screen_x, smoothed_screen_y = fn.smooth_mouse_position(screen_x, screen_y)
                mouse.position = (smoothed_screen_x * sensitivity_factor, smoothed_screen_y * sensitivity_factor)

                # Pinch detection to click
                pinch_threshold = 0.035 * image.shape[1]

                if distances[('THUMB_TIP', 'INDEX_FINGER_TIP')] < pinch_threshold:
                    dot_color = (255, 0, 0)
                    if not mouse_pressed:
                        mouse.press(Button.left)
                        mouse_pressed = True
                else:
                    dot_color = (0, 255, 0)
                    if mouse_pressed:
                        mouse.release(Button.left)
                        mouse_pressed = False

                # Draw a circle at the midpoint of thumb and index finger
                cv2.circle(image, midpoints[('THUMB_TIP', 'INDEX_FINGER_TIP')], 9, dot_color, -1)

        # Display the resulting image.
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
            break

cap.release()
cv2.destroyAllWindows()
