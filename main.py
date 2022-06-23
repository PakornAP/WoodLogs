import cv2 as cv
import numpy as np
from detection.find_contors import HSV_method, draw, Find_contours, drawff


def Resize(path, dim):
    img = cv.imread(path)
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return img


def Show_Frame(label: str, frame):
    cv.imshow(label, frame)
    if cv.waitKey(0) & 0xFF == ord("q"):
        cv.destroyAllWindows()


def Hog_circle(frame):
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (7, 7), 0)
    circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1.4, 100)
    output = frame.copy()
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv.circle(output, (x, y), r, (255, 0, 0), 2)
    return output


if __name__ == "__main__":
    path = "./dataset/1.jpg"
    # path = r"dataset\1655358724571.jpeg"
    dim = (720, 720)  # (width,height)
    frame = Resize(path, dim)
    # contours = Hog_circle(frame)
    # contours = drawff(frame, (Find_contours(frame)))
    # contours = draw(frame, Find_contours(frame))
    contours = draw(frame, HSV_method(frame))
    # Show_Frame('image', frame)
    Show_Frame("contours", contours)
