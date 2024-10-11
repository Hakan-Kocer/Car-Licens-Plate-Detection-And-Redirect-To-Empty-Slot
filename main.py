import cv2
import detection
import read
import pyautogui

# Open camera feed
input_cam = cv2.VideoCapture(2)
is_running = True

while is_running:
    # Prompt the user for vehicle status
    result = pyautogui.prompt(text='Press "y" if a vehicle has arrived, or "e" to exit', title='Vehicle Check')
    
    if result == "y":
        ret, img = input_cam.read()
        if ret:
            img = detection.detection(img)
            words = read.find_character(img, img)
            input_cam.release()
            plate.slot()
    elif result == "e":
        is_running = False

# Release resources
input_cam.release()
cv2.destroyAllWindows()
