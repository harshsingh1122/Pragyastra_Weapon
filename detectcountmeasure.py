import cv2
import numpy as np
import tkinter as tk
import Jetson.GPIO as GPIO
import time
from adafruit_servokit import ServoKit
import threading

import simpleaudio as sa


#~~~~~~~~~~~~~~~~~~~~~~~~~~~``
import tkinter as tk
from multiprocessing import Process
#from gallery_app import ImageGal leryApp
#--------------------------------
 
# Cleanup GPIO
GPIO.cleanup()

# Load MobileNet-SSD
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

# Load class labels
classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

KNOWN_DISTANCE = 2.0  # Distance from the camera to object in meters (known at the first frame)
KNOWN_WIDTH = 0.4     # Actual width of the object in meters (needs to be known in advance)
FOCAL_LENGTH = 615

# Loading video
cap = cv2.VideoCapture(0)

# Setup GPIO pins for stepper motor
DIR = 17
STEP = 18
CW = 0
CCW = 1
SPR = 90
MS1 = 22
MS2 = 23
MS3 = 24

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(DIR, GPIO.OUT)
GPIO.setup(STEP, GPIO.OUT)
GPIO.setup(MS1, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(MS2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(MS3, GPIO.OUT, initial=GPIO.HIGH)
# Setup GPIO pin for person detection







step_count = SPR
delay = 0.0005
current_pos = 0

# Setup PCA9685 for Servo Motors
kit = ServoKit(channels=16)
last_capture_time = 0  # This will store the time of the last capture

# Load your audio file (change path as needed)
sound_path = '/home/anikait/Documents/project_codes/SSD (Single Shot MultiBox Detector)_human_decetion/attack2t22wav-14511.wav'
wave_obj = sa.WaveObject.from_wave_file(sound_path)
play_obj = None

def start_sound():
    global play_obj
    if play_obj is None or not play_obj.is_playing():
        play_obj = wave_obj.play()

def stop_sound():
    global play_obj
    if play_obj is not None and play_obj.is_playing():
        play_obj.stop()

# Create a function to move servo 1 to 90 degrees and back
def move_servo_1():
    kit.servo[0].angle = 90
    time.sleep(1)
    kit.servo[0].angle = 0

"""def capture_image():
    global frame
    # check if frame is not None
    if frame is not None:
        # Generate a filename by the current time
        filename = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(filename + '.jpg', frame)"""
import os
path = "/home/anikait/Documents/project_codes/SSD (Single Shot MultiBox Detector)_human_decetion"
def open_folder():
    os.system(f'nautilus "{path}"')  # This will work for distributions using GNOME/Nautilus as the file manager

def start_gui():
    root = tk.Tk()
    root.title("Servo Control")
    root.geometry("300x200")

    # Dark theme colors
    bg_color = "#000000"
    fg_color = "#FFFFFF"
    button_bg_color = "#800080"
    root.configure(bg=bg_color)
    label_font = ("Helvetica", 16)  # Specify font family and size
    label_servo_1 = tk.Label(root, text="Main Screen", bg=bg_color, fg=fg_color, font=label_font)
    
    button_fire = tk.Button(root, text="Fire!", bg=button_bg_color, fg=fg_color, command=move_servo_1)
    #button_capture = tk.Button(root, text="Capture Image", bg=button_bg_color, fg=fg_color, command=capture_image)
    #button_gallery = tk.Button(root, text="Captured Photos", command=open_folder, bg=button_bg_color, fg=fg_color)
    label_servo_1.pack(pady=10)
    button_fire.pack(pady=10)
    #button_capture.pack(pady=10)
    #button_gallery.pack(pady=10)

    # Create a menu bar
    menu_bar = tk.Menu(root)
    
    
    # Add file menu
    file_menu = tk.Menu(menu_bar, tearoff=0)
    file_menu.add_command(label="Exit", command=root.quit)
    menu_bar.add_cascade(label="File", menu=file_menu)
    
    # Add about  menu
    help_menu = tk.Menu(menu_bar, tearoff=0)
    help_menu.add_command(label="About", command=show_about)
    menu_bar.add_cascade(label="About", menu=help_menu)

    # Add help menu
    help_menu = tk.Menu(menu_bar, tearoff=0)
    help_menu.add_command(label="Help", command=show_about)
    menu_bar.add_cascade(label="Help", menu=help_menu)
    
    root.config(menu=menu_bar)


    root.mainloop()
from threading import Thread



def show_about():
    about_window = tk.Toplevel()
    about_window.title("About")
    about_window.geometry("300x150")

    about_label = tk.Label(about_window, text="This is the Servo Control App.\nVersion 1.0", padx=10, pady=10)
    about_label.pack()



gui_thread = threading.Thread(target=start_gui)
gui_thread.start()

while True:
    ret, frame = cap.read()
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (150, 150)), 0.007843, (150, 150), 127.5)
    net.setInput(blob)
    detections = net.forward()

    h, w = frame.shape[:2]

    closest_person_details = None  # This will store details of the closest person

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # If a person is detected
                
                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                centerX, centerY = (startX + endX) // 2, (startY + endY) // 2
                person_location = (centerX, centerY)


           
                # Distance estimation
                pixel_width = endX - startX
                distance = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

                # If this person is closer than the currently closest person, or if it's the first person detected
                if closest_person_details is None or distance < closest_person_details['distance']:
                    closest_person_details = {
                        'distance': distance,
                        'person_location': person_location,
                        'box': (startX, startY, endX, endY)
                    }
                else:
                # This person is not the closest, so draw a blue box
                 cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            #else:
                

    if closest_person_details is not None:
        start_sound()
        distance = closest_person_details['distance']
        person_location = closest_person_details['person_location']
        (startX, startY, endX, endY) = closest_person_details['box']

        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, f"Distance: {distance:.2f}m, Coords: {person_location}", (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        distance_from_center = person_location[0] - w // 2

        if abs(distance_from_center) > 50:
            # Person is not in the center, move the servo to 0 degrees (manual mode)
            kit.servo[0].angle = 0
            servo_automatic_mode = False
        else:
            # Person is in the center, move the servo to 90 degrees (automatic mode)
            if not servo_automatic_mode:
                kit.servo[0].angle = 90
                servo_automatic_mode = True
            


    else:
        stop_sound()
        last_capture_time = 0  # Reset the capture time if no person is detected

        if current_pos != 0:
            direction = CW if current_pos < 0 else CCW
            GPIO.output(DIR, direction)
            for x in range(abs(current_pos)):
                GPIO.output(STEP, GPIO.HIGH)
                time.sleep(delay)
                GPIO.output(STEP, GPIO.LOW)
                time.sleep(delay)
            current_pos = 0

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
