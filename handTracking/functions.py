import numpy as np

# EMA Smoothing
landmark_smoothing_factor = 0.1
mouse_smoothing_factor = 0.2
previous_coordinates = {}
previous_mouse_x, previous_mouse_y = None, None


def smooth_landmarks(current_coordinates):
    global previous_coordinates
    smoothed_coordinates = {}
    for key in current_coordinates:
        if key in previous_coordinates:
            smoothed_coordinates[key] = landmark_smoothing_factor * np.array(current_coordinates[key]) + (
                        1 - landmark_smoothing_factor) * np.array(previous_coordinates[key])
        else:
            smoothed_coordinates[key] = np.array(current_coordinates[key])
    previous_coordinates = smoothed_coordinates
    return smoothed_coordinates


def get_landmark_coordinates(landmarks, image_shape, mp_hands):
    coordinates = {}
    for landmark in ['THUMB_TIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_TIP', 'PINKY_TIP']:
        coord = landmarks.landmark[getattr(mp_hands.HandLandmark, landmark)]
        coordinates[landmark] = (int(coord.x * image_shape[1]), int(coord.y * image_shape[0]))
    return coordinates


def smooth_mouse_position(mouse_x, mouse_y):
    global previous_mouse_x, previous_mouse_y
    if previous_mouse_x is None or previous_mouse_y is None:
        previous_mouse_x, previous_mouse_y = mouse_x, mouse_y
    smoothed_mouse_x = mouse_smoothing_factor * mouse_x + (1 - mouse_smoothing_factor) * previous_mouse_x
    smoothed_mouse_y = mouse_smoothing_factor * mouse_y + (1 - mouse_smoothing_factor) * previous_mouse_y
    previous_mouse_x, previous_mouse_y = smoothed_mouse_x, smoothed_mouse_y
    return smoothed_mouse_x, smoothed_mouse_y


def calculate_midpoints(coordinates):
    midpoints = {}
    pairs = [('THUMB_TIP', 'INDEX_FINGER_TIP'), ('THUMB_TIP', 'MIDDLE_FINGER_TIP'),
             ('THUMB_TIP', 'RING_FINGER_TIP'), ('THUMB_TIP', 'PINKY_TIP')]
    for pair in pairs:
        mid_x = (coordinates[pair[0]][0] + coordinates[pair[1]][0]) // 2
        mid_y = (coordinates[pair[0]][1] + coordinates[pair[1]][1]) // 2
        midpoints[pair] = (mid_x, mid_y)
    return midpoints


def calculate_distances(coordinates):
    distances = {}
    pairs = [('THUMB_TIP', 'INDEX_FINGER_TIP'), ('THUMB_TIP', 'MIDDLE_FINGER_TIP'),
             ('THUMB_TIP', 'RING_FINGER_TIP'), ('THUMB_TIP', 'PINKY_TIP')]
    for pair in pairs:
        distance = np.sqrt((coordinates[pair[0]][0] - coordinates[pair[1]][0]) ** 2 +
                           (coordinates[pair[0]][1] - coordinates[pair[1]][1]) ** 2)
        distances[pair] = distance
    return distances
