import cv2 as cv
import numpy as np


def Resize(path, dim):
    img = cv.imread(path)
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return img


def Show_Frame(label: str, frame):
    cv.imshow(label, frame)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return frame


def Hog_circle(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # frame = cv.GaussianBlur(frame, (7, 7), 0)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1.4, 100)
    output = frame.copy()
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv.circle(output, (x, y), r, (255, 0, 0), 4)
            cv.rectangle(output, (x-5, y-5), (x+5, y+5), (0, 128, 255), -1)
    return output


if __name__ == '__main__':
    # path = './dataset/cap.jpeg'
    path = 'dataset/1655358792384.jpeg'
    dim = (640, 640)  # (width,height)
    frame = Resize(path, dim)
    contours = Hog_circle(frame)
    # Show_Frame('image', frame)
    Show_Frame('contours', contours)
