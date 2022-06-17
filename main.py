import cv2 as cv


def Resize(path, dim):
    img = cv.imread(path, cv.IMREAD_COLOR)
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return img


def Show_Frame(label: str, frame):
    cv.imshow(label, frame)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return frame


def Find_contours(frame):
    # img pre-processing
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.GaussianBlur(frame, (7, 7), 0)
    ret, frame = cv.threshold(frame, 100, 255, cv.THRESH_BINARY)
    cnts, hierarchy = cv.findContours(
        image=frame, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    cnts = cv.drawContours(image=frame.copy(), contours=cnts, contourIdx=-1,
                           color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
    return cnts


if __name__ == '__main__':
    path = './dataset/1655358724571.jpeg'
    dim = (640, 640)  # (width,height)
    frame = Resize(path, dim)
    contours = Find_contours(frame)
    Show_Frame('frame', contours)
