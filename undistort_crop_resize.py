#!/usr/bin/env python

# You should replace these 3 lines with the output in calibration step

import numpy as np
import sys
import cv2
import glob
import os


DIM=(640, 480)
# can also be read from cam_info.yaml file
K=np.array([[399.899682021, 0.0, 314.361624631], [0.0, 403.092496868, 231.601274145], [0.0, 0.0, 1.0]])
D=np.array([[-0.0968820835317], [0.0325666489702], [-0.0848936174991], [0.0769170966581]])


def undistort_setup(dim1=(640, 480), balance=0.93, dim2=None, dim3=None):
    # img = cv2.imread(img_path)
    # dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort

    # print dim1
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)

    # keep center of image fixed
    new_K[0][2] = K[0][2]
    new_K[1][2] = K[1][2]

    # print new_K


    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)

    return map1, map2
    # undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # cv2.imshow("before", img)
    # cv2.imshow("undistorted", undistorted_img)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def undistort(img_path, map1, map2):
    img = cv2.imread(img_path)
    # convert to gray image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.remap(gray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


if __name__ == '__main__':
    images = glob.glob('../fingercam/*/*.jpg')
    map1, map2 = undistort_setup()

    for img in images:
        s1, s2, s3, s4 = img.split('/')
        s2 = s2 + '_undistorted'
        new_path = os.path.join(s1, s2, s3, s4)
        # print 'undistorting...'

        # img = '../fingercam/init/img_init.jpg'
        image_undistorted = undistort(img, map1, map2)

        #grop to 440X440
        image_c = image_undistorted[0 + 42:480 - 42, 120 - 75:520 + 75]  # 440X440
        # print 'resizing...'
        # cv2.imshow('checking size', image_c)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # break

        image_small = cv2.resize(image_c, (0, 0), fx=1, fy=1)  # resize to half size -> 330X330
        #print 'saving to path: ' + new_path
        cv2.imwrite(new_path, image_small)

