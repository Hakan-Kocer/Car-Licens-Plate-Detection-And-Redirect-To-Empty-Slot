import cv2
import numpy as np
from drawing_utils import draw_contours

COLOR_WHITE = (255, 255, 255)
COLOR_BLUE = (255, 0, 0)

class CoordinatesGenerator:
    KEY_RESET = ord("r")
    KEY_QUIT = ord("q")

    def __init__(self, image, output):
        self.output = output
        self.caption = image
        self.color = COLOR_BLUE

        self.image = cv2.imread(image).copy()
        self.click_count = 0
        self.ids = 0
        self.coordinates = []

        cv2.namedWindow(self.caption, cv2.WINDOW_GUI_EXPANDED)
        cv2.setMouseCallback(self.caption, self.__mouse_callback)

    def generate(self):  # Checks if a key is pressed
        while True:
            cv2.imshow(self.caption, self.image)
            key = cv2.waitKey(0)

            if key == CoordinatesGenerator.KEY_RESET:
                self.image = self.image.copy()
            elif key == CoordinatesGenerator.KEY_QUIT:
                break

        cv2.destroyWindow(self.caption)

    def __mouse_callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:  # When clicked on any location
            self.coordinates.append((x, y))  # Saves the clicked positions
            self.click_count += 1  # Checks how many times clicked

            if self.click_count >= 4:
                self.__handle_done()
            elif self.click_count > 1:
                self.__handle_click_progress()

        cv2.imshow(self.caption, self.image)

    def __handle_click_progress(self):  # Draws lines between points if more than one click
        cv2.line(self.image, self.coordinates[-2], self.coordinates[-1], COLOR_BLUE, 1)

    def __handle_done(self):  # Draws a square if four points are clicked
        cv2.line(self.image, self.coordinates[2], self.coordinates[3], self.color, 1)
        cv2.line(self.image, self.coordinates[3], self.coordinates[0], self.color, 1)

        self.click_count = 0
        coordinates = np.array(self.coordinates)

        self.output.write("-\n          id: " + str(self.ids) + "\n          coordinates: [" +
                          "[" + str(self.coordinates[0][0]) + "," + str(self.coordinates[0][1]) + "]," +
                          "[" + str(self.coordinates[1][0]) + "," + str(self.coordinates[1][1]) + "]," +
                          "[" + str(self.coordinates[2][0]) + "," + str(self.coordinates[2][1]) + "]," +
                          "[" + str(self.coordinates[3][0]) + "," + str(self.coordinates[3][1]) + "]]\n")

        draw_contours(self.image, coordinates, str(self.ids + 1), COLOR_WHITE)

        for _ in range(4):
            self.coordinates.pop()

        self.ids += 1
