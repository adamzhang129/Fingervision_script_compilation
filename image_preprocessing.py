import cv2
import os
import numpy as np

img1 = cv2.imread(os.path.join('../fingercam_undistorted/init/', 'img_init.jpg'), 0)
# img2 = cv2.imread(os.path.join('./fingercam/displacement/', '0003.jpg'), 0)
# img1 = cv2.imread(os.path.join('./fingercam/displacement/', '0116.jpg'), 0)

img2 = cv2.imread(os.path.join('../fingercam_undistorted/normal/', '0120.jpg'), 0)


def preprocessing(img):
    # cv2.imshow('before', img)
    # gaussian blurring
    img = cv2.GaussianBlur(img, (3, 3), 0)
    white = np.ones((img.shape[0], img.shape[1]))
    thresh = white - cv2.threshold(img, 65, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((3, 3), np.uint8)

    dilated = cv2.dilate(thresh, kernel)

    # cv2.imshow('dilated first', dilated)

    eroded = cv2.erode(dilated, np.ones((6, 6), np.uint8))

    # cv2.imshow('eroded', eroded)

    # thresh = eroded

    # kernel = np.ones((5,5), np.uint8)
    # opening = cv2.erode(thresh, kernel)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('opened', opening)

    kernel = np.ones((5, 5), np.uint8)

    dilated = cv2.dilate(eroded, kernel)

    # cv2.imshow('dilated', dilated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return dilated


if __name__ == '__main__':
    img_proced = preprocessing(img2)
    cv2.imshow('image processed', img_proced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
