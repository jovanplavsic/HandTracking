import cv2
import mediapipe as mp
from pynput.mouse import Controller
import tkinter as tk

# Initialize MediaPipe Hands.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Create a VideoCapture object.
cap = cv2.VideoCapture(0)
image = None
hand_landmarks = None

# Create a mouse controller object.
mouse = Controller()

# Get screen size using Tkinter.
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.withdraw()  # Hide the Tkinter root window


