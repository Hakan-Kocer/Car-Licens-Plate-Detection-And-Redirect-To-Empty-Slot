import cv2
import numpy as np
import pyautogui

# Define color constants
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)

empty_spots = []
occupied_spots = []

class MotionDetector:
    LAPLACIAN = 1.4
    DETECT_DELAY = 1

    def __init__(self, video, coordinates, start_frame):
        self.video = video
        self.coordinates_data = coordinates
        self.start_frame = start_frame
        self.contours = []
        self.bounds = []
        self.mask = []
        self.status_list = []

    def detect_motion(self):
        cv2.imread(self.video)

        for p in self.coordinates_data:
            coordinates = self._coordinates(p)
            rect = cv2.boundingRect(coordinates)  # Find the smallest rectangle that contains all points

            new_coordinates = coordinates.copy()
            new_coordinates[:, 0] -= rect[0]
            new_coordinates[:, 1] -= rect[1]

            self.contours.append(coordinates)
            self.bounds.append(rect)

            mask = cv2.drawContours(
                np.zeros((rect[3], rect[2]), dtype=np.uint8),
                [new_coordinates],
                contourIdx=-1,
                color=255,
                thickness=-1,
                lineType=cv2.LINE_8
            )

            mask = mask == 255
            self.mask.append(mask)

        statuses = [False] * len(self.coordinates_data)
        times = [None] * len(self.coordinates_data)
        i = 0
        status_list = []
        j = 0
        frame = cv2.imread(self.video)

        blurred = cv2.GaussianBlur(frame.copy(), (5, 5), 3)
        grayed = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        if j == 0:
            j = 1
            for index, _ in enumerate(self.coordinates_data):
                occupied_spots.append(index + 1)
                empty_spots.append(index + 1)
            print(occupied_spots)
            print(empty_spots)
            
        for index, c in enumerate(self.coordinates_data):
            status = self.__apply(grayed, index, c)

            if i == 0:
                status_list.append(status)
                if not status:
                    print(f"{index + 1}. parking spot is occupied")
                    a = index + 1
                    empty_spots.remove(a)
                else:
                    print(f"{index + 1}. parking spot is empty")
                    a = index + 1
                    occupied_spots.remove(a)

            if status_list[index] != status:
                if not status:
                    print(f"{index + 1}. parking spot is occupied")
                    a = index + 1
                    occupied_spots.append(a)
                    empty_spots.remove(a)
                else:
                    print(f"{index + 1}. parking spot is empty")
                    a = index + 1
                    empty_spots.append(a)
                    occupied_spots.remove(a)

                status_list[index] = status

        if i == 0:
            i = 1

        cv2.destroyAllWindows()
        min_index = np.argmin(empty_spots)
        print(f"You can move to parking spot number {empty_spots[min_index]}")
        pyautogui.alert(
            text=f"You can move to parking spot number {empty_spots[min_index]}",
            title='Available Spot',
            button='OK'
        )

    def __apply(self, grayed, index, p):
        coordinates = self._coordinates(p)
        rect = self.bounds[index]
        roi_gray = grayed[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
        laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)

        coordinates[:, 0] -= rect[0]
        coordinates[:, 1] -= rect[1]

        status = np.mean(np.abs(laplacian * self.mask[index])) < MotionDetector.LAPLACIAN
        return status

    @staticmethod
    def _coordinates(p):
        return np.array(p["coordinates"])


class CaptureReadError(Exception):
    pass
