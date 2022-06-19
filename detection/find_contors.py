import cv2


def Find_contours(frame):
    # img pre-processing
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.GaussianBlur(frame, (7, 7), 0)
    ret, frame = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)
    cnts, hierarchy = cv2.findContours(
        image=frame, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cv2.drawContours(image=frame.copy(), contours=cnts, contourIdx=-1,
    #                         color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    return cnts


def HSV_method(roi):
    # reduce noise
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    # hsv = cv2.GaussianBlur(hsv, (15, 15), 0)
    _, bw = cv2.threshold(hsv[:, :, 2], 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    dist = cv2.distanceTransform(morph, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    borderSize = 30
    distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize,
                                    cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    gap = 10
    kernel2 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*(borderSize-gap)+1, 2*(borderSize-gap)+1))
    kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap,
                                 cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    distTempl = cv2.distanceTransform(
        kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, _ = cv2.minMaxLoc(nxcor)
    _, peaks = cv2.threshold(nxcor, mx*0.45, 255, cv2.THRESH_BINARY)
    peaks8u = cv2.convertScaleAbs(peaks)
    # cv2.imshow('test', peaks)
    contours, _ = cv2.findContours(
        peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('processing', peaks8u)
    return contours


def draw(frame, cnts):
    for cnt in cnts:  # each of contours
        area = cv2.contourArea(cnt)
        if area <= 8:
            continue
        print('Area : ', area)
        ellipse = cv2.fitEllipse(cnt)  # (x,y),(a,b),angle
        cv2.ellipse(frame, ellipse, (255, 100, 2), 2)
        # current.append([x, y, a, b])
        # print(f'current : {current}')
    return frame
