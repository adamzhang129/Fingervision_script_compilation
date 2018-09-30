from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import threading
import time

from undistort_crop_resize import *
from sklearn.neighbors import KDTree
import cv2
import numpy as np

import pandas as pd
pd.set_option('display.max_rows', 500)


def crop(img):
    y_min = 20
    y_max = 480 - 20
    x_min = 120 - 60
    x_max = 520 + 60

    return img[y_min:y_max, x_min:x_max]


def blob_detector_setup():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()


    params.filterByColor = 1
    params.blobColor = 0
    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 60

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.8

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    return detector



def blob_detect(detector, img):
    keypoints = detector.detect(img)

    # print(len(keypoints))
    locs = []
    for i in range(0, len(keypoints)):
        locs.append([keypoints[i].pt[0], keypoints[i].pt[1]])

    # print(np.array(locs))
    return np.array(locs), keypoints


def tracking_thread(cap, map1, map2, blob_detector):
    t = threading.current_thread()
    count = 0

    loc_0 = []
    while getattr(t, 'keep_running', True):
        # Capture frame-by-frame
        # print('testing')
        ret, frame = cap.read()

        # cv2.imshow('name', frame)
        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray_proc = preprocess(gray)
        gray_ud = cv2.remap(gray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        gray_crop = crop(gray_ud)

        loc, keypoints = blob_detect(blob_detector, gray_crop)
        # print(loc.shape)
        # print(loc)
        # im_with_keypoints = cv2.drawKeypoints(gray_crop, keypoints, np.array([]),
        #                                       (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow('keypoints', im_with_keypoints)

        if count == 0:
            loc_0 = loc.copy()
            recent_loc = loc.copy()
        elif count > 0:
            # print('============================')
            # print(loc_0[1,:])
            kdt = KDTree(loc, leaf_size=30, metric='euclidean')
            dist, ind = kdt.query(recent_loc, k=1)
            thd = (dist < 14)*1
            thd_nz = np.where(thd)[0]
            # update point if close enough point are detected
            recent_loc[thd_nz] = np.reshape(loc[ind[thd_nz]], (len(thd_nz), 2))
            print('{} points updated'.format(len(thd_nz)))
            # visualize the displacement field
            loc_v = 2*recent_loc - loc_0  # diff vector
            # for i in range(0, len(loc_0)):
            #     cv2.arrowedLine(gray_crop, (int(np.around(recent_loc[i, 0])), int(np.around(recent_loc[i, 1]))),
            #                     (int(np.around(loc_v[i, 0])), int(np.around(loc_v[i, 1]))), (0, 0, 255), thickness=2)


            for i in range(0, len(loc_0)):
                cv2.arrowedLine(gray_crop, (int(np.around(recent_loc[i, 0])), int(np.around(recent_loc[i, 1]))),
                                (int(np.around(loc_v[i, 0])), int(np.around(loc_v[i, 1]))), (0, 0, 255), thickness=2)

            if getattr(t, 'save', False):
                print('saving end location and location')
                cv2.imwrite('./images/image.jpg', gray_crop)
                setattr(t, 'save', False)  # toggle to save just once
            # cv2.imshow('arrow', gray_crop)
            # print(gray_crop.shape)


        count += 1

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    print('tracking thread stopped')
    # cv2.destroyAllWindows()


def main():
    cap = cv2.VideoCapture(0)
    # undistort_setup()
    map1, map2 = undistort_setup()
    blob_detector = blob_detector_setup()

    while True:

        t = threading.Thread(target=tracking_thread, args=(cap, map1, map2, blob_detector))
        # t.keep_running = True

        t.start()
        time.sleep(6)
        t.save = True
        time.sleep(2)
        t.keep_running = False
        t.join()
        print('sleeping for 3 sec')
        time.sleep(3)

if __name__ == "__main__":
    main()