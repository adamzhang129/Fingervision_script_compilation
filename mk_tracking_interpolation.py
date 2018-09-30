from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

from undistort_crop_resize import *

from sklearn.neighbors import KDTree

import cv2

import numpy as np

import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)


from scipy import ndimage
from skimage import morphology, util, filters
import skimage



cap = cv2.VideoCapture(0)
undistort_setup()
map1, map2 = undistort_setup()


def crop(img):
    y_min = 20
    y_max = 480 - 20
    x_min = 120 - 80
    x_max = 520 + 80

    return img[y_min:y_max, x_min:x_max]


def preprocess(img):
    """
    Apply image processing functions to return a binary image
    """

    # Apply thresholds
    cv2.imshow('unprocessed', img)
    img = filters.threshold_local(img, 3)
    threshold = 0.3
    idx = img > img.max() * threshold
    idx2 = img < img.max() * threshold
    img[idx] = 0
    img[idx2] = 255

    # undistorting
    img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Crop the pictures as for raw images.
    img = crop(img)
    cv2.imshow('proced', img)


    struct = ndimage.generate_binary_structure(2, 3)
    # img = ndimage.binary_dilation(img, structure=struct)
    img = ndimage.binary_erosion(img, ndimage.generate_binary_structure(2, 9))
    img = ndimage.binary_dilation(img, structure=struct)

    # cv2.imshow('proced', util.img_as_int(img))

    return util.img_as_int(img)


def feature_extract(img):
    feature = pd.DataFrame()
    # for num, img in enumerate(frames):
    label_image = skimage.measure.label(img)
    # flip color
    white = np.ones((img.shape[0], img.shape[1]))
    img = white - img
    # print(len(skimage.measure.regionprops(label_image, intensity_image=img)))
    # count = 0
    for region in skimage.measure.regionprops(label_image, intensity_image=img):
        # Everywhere, skip small and large areas
        # print(region.area, region.mean_intensity)
        # print(count)

        if region.area < 20 or region.area > 800:
            continue
        # Only black areas
        if region.mean_intensity > 1:
            continue
        # print('check if come here')
        # On the top, skip small area with a second threshold
        #         if region.centroid[0] < 260 and region.area < 80:
        #             continue
        # Store features which survived to the criterions
        feature = feature.append([{'y': region.centroid[0],
                                   'x': region.centroid[1],
                                   'frame': 1,
                                   }, ])

    return feature


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



def blob_detect(img):
    keypoints = detector.detect(img)

    # print(len(keypoints))
    locs = []
    for i in range(0, len(keypoints)):
        locs.append([keypoints[i].pt[0], keypoints[i].pt[1] ] )

    # print(np.array(locs))
    return np.array(locs), keypoints



count = 0

loc_0 = []
while (True):
    # Capture frame-by-frame
    # print('testing')
    ret, frame = cap.read()

    # operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray_proc = preprocess(gray)
    gray_ud = cv2.remap(gray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    gray_crop = crop(gray_ud)

    # print(gray_crop.shape)
    # save the image
    cv2.imwrite('./tracking/image1.jpg', gray_crop)

    loc, keypoints = blob_detect(gray_crop)
    # print(loc.shape)
    # print(loc)
    # im_with_keypoints = cv2.drawKeypoints(gray_crop, keypoints, np.array([]),
    #                                       (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow('keypoints', im_with_keypoints)

    if count == 0:

        loc_0 = loc.copy()
        recent_loc = loc.copy()
    elif count > 0:
        print('===========frame: {}================='.format(count))
        # print(loc_0[1,:])
        kdt = KDTree(loc, leaf_size=30, metric='euclidean')
        dist, ind = kdt.query(recent_loc, k=1)
        thd = (dist < 14)*1
        thd_nz = np.where(thd)[0]
        # update point if close enough point are detected
        recent_loc[thd_nz] = np.reshape(loc[ind[thd_nz]], (len(thd_nz), 2))

        # visualize the displacement field
        loc_v = 2*recent_loc - loc_0  # diff vector

        img_rgb = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB)
        # draw image and save vectors
        for i in range(0, len(loc_0)):
            cv2.arrowedLine(img_rgb, (int(np.around(recent_loc[i, 0])), int(np.around(recent_loc[i, 1]))),
                            (int(np.around(loc_v[i, 0])), int(np.around(loc_v[i, 1]))), (0, 0, 255), thickness=2)
        cv2.imshow('arrow', img_rgb)


        df = pd.DataFrame(np.concatenate((recent_loc, loc_v), axis=1), columns=['x', 'y', 'xt', 'yt'])
        df.to_csv('./tracking/vectors1.csv')

    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
