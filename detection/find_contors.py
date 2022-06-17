import cv2 as cv


def Find_contours(frame):
    # img pre-processing
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.GaussianBlur(frame, (7, 7), 0)
    ret, frame = cv.threshold(frame, 100, 255, cv.THRESH_BINARY)
    cnts, hierarchy = cv.findContours(
        image=frame, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    cnts = cv.drawContours(image=frame.copy(), contours=cnts, contourIdx=-1,
                           color=(0, 255, 0), thickness=1, lineType=cv.LINE_AA)
    return cnts
