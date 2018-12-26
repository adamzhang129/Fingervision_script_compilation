# from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

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
        locs.append([keypoints[i].pt[0], keypoints[i].pt[1]])

    # print(np.array(locs))
    return np.array(locs), keypoints


from scipy.interpolate import Rbf

count = 0

loc_0 = []

df1 = pd.DataFrame(columns=['Fx', 'Fy', 'Tz'])

import time

Nx = 30
Ny = 30
xx = np.linspace(0, 560, Nx)
yy = np.linspace(0, 440, Ny)

XX, YY = np.array(np.meshgrid(xx, yy))

XXX = XX.ravel()  # flatten
YYY = YY.ravel()
# --- decomposition setup ------
from pynhhd.nHHD import nHHD

dx = 560.0 / (Nx - 1)
dy = 440.0 / (Ny - 1)
grid = (Ny, Nx)

decomposition_obj = nHHD(grid=grid, spacings=(dy, dx))

# ----------------------------
white_img = np.zeros([440, 560, 3], dtype=np.uint8)
white_img.fill(255)


def wrench_estimate(X_, Y_, d, r, R, h):
    # inputs: (all in column vector form (Nx*Ny, 2))
    #       d: curl free field
    #       r: divergence free field
    #       h: harmonic field

    mag_d = np.hypot(np.abs(d[:, 0]), np.abs(d[:, 1]))
    sum_d = np.sum(mag_d, axis=0)
    # print 'total magnitude of d: {}'.format(sum_d)

    # calculate the index of maximum and minimum of potential field as rotation center
    rot_max_ind = np.argmax(R)
    rot_max_loc = (X_[rot_max_ind], Y_[rot_max_ind])

    rot_min_ind = np.argmin(R)
    rot_min_loc = (X_[rot_min_ind], Y_[rot_min_ind])

    # validate using first derivative
    X = X_.reshape(Nx, Ny)
    Y = Y_.reshape(Nx, Ny)

    i_max = rot_max_ind // Nx
    j_max = rot_max_ind % Nx

    i_min = rot_min_ind // Nx
    j_min = rot_min_ind % Nx

    # validate with finite difference symbols
    if i_max > 0 and i_max <29 and j_max >0 and j_max <29:
        if (R[i_max, j_max] - R[i_max - 1, j_max]) > 0 and (R[i_max + 1, j_max] - R[i_max, j_max]) < 0:
            if (R[i_max, j_max] - R[i_max, j_max - 1]) > 0 and (R[i_max, j_max + 1] - R[i_max, j_max]) < 0:
                # this point is the local minima
                print 'max exist'
                Max = rot_max_loc
            else:

                Max = None
        else:
            Max = None
    else:
        Max =  None

    if i_min > 0 and i_min < 29 and j_min > 0 and j_min < 29:
        if (R[i_min, j_min] - R[i_min - 1, j_min]) < 0 and (R[i_min + 1, j_min] - R[i_min, j_min]) > 0:
            if (R[i_min, j_min] - R[i_min, j_min - 1]) < 0 and (R[i_min, j_min + 1] - R[i_min, j_min]) > 0:
                # this point is the local minima
                print 'min exist'
                Min = rot_min_loc
            else:

                Min = None
        else:
            print 'min does not exist'
            Min = None
    else:
        Min = None

    XY = np.stack((X_, Y_), axis=1)
    if Max != None:
        rot_diff = XY - rot_max_loc
        torque = np.cross(rot_diff, r)
        torque_max = np.sum(torque, axis=0)
    else:
        torque_max = 0.0

    if Min != None:
        rot_diff = XY - rot_min_loc
        torque = np.cross(rot_diff, r)
        torque_min = np.sum(torque, axis=0)
    else:
        torque_min = 0.0

    sum_torque = (torque_max + torque_min)/(Nx*Ny)


    # sum up the harmonic field
    sum_h = np.sum(h, axis=0)
    sum_all = np.sum(d+r+h, axis=0)

    return sum_d, sum_torque, sum_h, sum_all






while (True):

    start = time.time()
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
    cv2.imwrite('./tracking/image.jpg', gray_crop)

    loc, keypoints = blob_detect(gray_crop)
    # print(loc.shape)
    # print(loc)
    # im_with_keypoints = cv2.drawKeypoints(gray_crop, keypoints, np.array([]),
    #                                       (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow('keypoints', im_with_keypoints)
    # cv2.imwrite('./tracking/image1.jpg', im_with_keypoints)

    # print 'time interval 1: {}'.format(time.time() - start)
    # last_time = time.time()

    if count == 0:

        loc_0 = loc.copy()
        recent_loc = loc.copy()
    elif count > 0:
        # print('============frame {}================'.format(count))
        # print(loc_0[1,:])
        kdt = KDTree(loc, leaf_size=30, metric='euclidean')
        dist, ind = kdt.query(recent_loc, k=1)
        thd = (dist < 14) * 1
        thd_nz = np.where(thd)[0]
        # update point if close enough point are detected
        recent_loc[thd_nz] = np.reshape(loc[ind[thd_nz]], (len(thd_nz), 2))

        # visualize the displacement field
        loc_v = 2 * recent_loc - loc_0  # diff vector

        # print 'time interval 2: {}'.format(time.time() - last_time)
        # last_time = time.time()

        # img_rgb = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB)
        # # draw image and save vectors
        # for i in range(0, len(loc_0)):
        #     cv2.arrowedLine(img_rgb, (int(np.around(recent_loc[i, 0])), int(np.around(recent_loc[i, 1]))),
        #                     (int(np.around(loc_v[i, 0])), int(np.around(loc_v[i, 1]))), (0, 0, 255), thickness=2)
        # cv2.imshow('arrow', img_rgb)
        # cv2.imwrite('./tracking/image2.jpg', img_rgb)

        # df = pd.DataFrame(np.concatenate((recent_loc, loc_v), axis=1), columns=['x', 'y', 'xt', 'yt'])
        # df.to_csv('./tracking/vectors.csv')

        # interpolation
        disp = recent_loc - loc_0

        dx, dy = disp[:, 0], disp[:, 1]
        x, y = recent_loc[:, 0], recent_loc[:, 1]

        interpolation_x = Rbf(x, y, dx)
        interpolation_y = Rbf(x, y, dy)

        dx_interp = interpolation_x(XXX, YYY)
        dy_interp = interpolation_y(XXX, YYY)
        mag = np.sqrt(dx_interp ** 2 + dy_interp ** 2)

        # print 'time interval 3: {}'.format(time.time() - last_time)
        # last_time = time.time()

        # img_rgb = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB)
        # # draw image and save vectors
        # for i in range(0, len(dx_interp)):
        #     cv2.arrowedLine(img_rgb, (int(np.around(XXX[i])), int(np.around(YYY[i]))),
        #                     (int(np.around(XXX[i] + dx_interp[i])), int(np.around(YYY[i] + dy_interp[i]))),
        #                     (0, 255, 255), thickness=2, tipLength=0.5)

        # cv2.imshow('arrow_interp', img_rgb)
        # print img_rgb.shape

        # df = pd.DataFrame(np.stack((XXX, YYY, dx_interp, dy_interp), axis=1), columns=['x', 'y', 'xt', 'yt'])
        # df.to_csv('./tracking/vectors_interp.csv')

        # cv2.imwrite('./tracking/image3.jpg', img_rgb)

        # print dx_interp.max(), dx_interp.min()

        # print 'time interval 4: {}'.format(time.time() - last_time)
        # last_time = time.time()

        vfield = np.stack((dx_interp.reshape(Nx, Ny), dy_interp.reshape(Nx, Ny)), axis=2)
        decomposition_obj.decompose(vfield, verbose=0)
        d = decomposition_obj.d
        h = decomposition_obj.h
        r = decomposition_obj.r
        # print d.shape

        d = d.reshape(Nx * Ny, 2)
        h = h.reshape(Nx * Ny, 2)
        r = r.reshape(Nx * Ny, 2)

        # print 'time interval 5: {}'.format(time.time() - last_time)
        # last_time = time.time()


        # esitmate normal force, torque and tangential force
        normal, torque, shear, sum_all = wrench_estimate(XXX, YYY, d, r, decomposition_obj.nRu, h)
        print normal, torque, shear, sum_all


        # paint on white image
        white = white_img.copy()
        for i in range(0, len(dx_interp)):
            cv2.arrowedLine(white, (int(np.around(XXX[i])), int(np.around(YYY[i]))),
                            (int(np.around(XXX[i] + d[i, 0])), int(np.around(YYY[i] + d[i, 1]))),
                            (0, 0, 0), thickness=2, tipLength=0.5)
        cv2.imshow('curl free field', white)

        white = white_img.copy()
        for i in range(0, len(dx_interp)):
            cv2.arrowedLine(white, (int(np.around(XXX[i])), int(np.around(YYY[i]))),
                            (int(np.around(XXX[i] + r[i, 0])), int(np.around(YYY[i] + r[i, 1]))),
                            (0, 0, 0), thickness=2, tipLength=0.5)
        cv2.imshow('divergence free field', white)

        white = white_img.copy()
        for i in range(0, len(dx_interp)):
            cv2.arrowedLine(white, (int(np.around(XXX[i])), int(np.around(YYY[i]))),
                            (int(np.around(XXX[i] + h[i, 0])), int(np.around(YYY[i] + h[i, 1]))),
                            (0, 0, 0), thickness=2, tipLength=0.5)
        cv2.imshow('harmonic field', white)

        # print 'time interval 6: {}'.format(time.time() - last_time)
        # last_time = time.time()

        # #-------------- vector field projections ---------
        # dx_resized = (dx_interp.reshape(Nx, Ny)+30)*256/60
        # dy_resized = (dy_interp.reshape(Nx, Ny)+30)*256/60
        # mag_resized = (mag.reshape(Nx, Ny) + 30)*256/60
        # # print '====== ', dx_resized.max(), dx_resized.min()
        #
        # fx = 6.2
        # fy = 6.2
        #
        # dx_large = cv2.resize(dx_resized, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        # dy_large = cv2.resize(dy_resized, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        # mag_large = cv2.resize(mag_resized, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        # # print dx_large.shape
        # # concatenate 3 images to fit 560 (the size of captured image)
        # black1 = np.ones((dx_large.shape[0], 1))*(-256)
        # black2 = np.ones((dx_large.shape[0], 1))*(-256)
        # concated = np.concatenate((dx_large, black1, dy_large, black2, mag_large), axis=1)
        # concated = np.array(concated, dtype=np.uint8)
        #
        # # dx_large = np.array(dx_large, dtype=np.uint8)
        # # dy_large = np.array(dy_large, dtype=np.uint8)
        # # mag_large = np.array(mag_large, dtype=np.uint8)
        #
        #
        #
        # # dx_large = cv2.applyColorMap(dx_large, cv2.COLORMAP_JET)
        # # dy_large = cv2.applyColorMap(dy_large, cv2.COLORMAP_JET)
        # # mag_large = cv2.applyColorMap(mag_large, cv2.COLORMAP_JET)
        #
        #
        # concated = cv2.applyColorMap(concated, cv2.COLORMAP_JET)
        #
        # # cv2.imshow('dx dy mag', concated)
        #
        #
        # # concatenate with captured image
        # images = np.concatenate((img_rgb, concated), axis=0)
        #
        # cv2.imshow('monitor window', images)

        # cv2.imshow('dx', dx_large)
        # cv2.imshow('dy', dy_large)
        # cv2.imshow('mag', mag_large)
        # cv2.imwrite('tracking/dx.jpg', dx_large)
        # cv2.imwrite('tracking/dy.jpg', dy_large)
        # cv2.imwrite('tracking/mag.jpg', mag_large)

        # # calculate center_x center_y fx fy and torque
        # # find the sum and average of dx_interp and dy_interp
        # sum_dx_i = np.sum(dx_interp)
        # sum_dy_i = np.sum(dy_interp)
        # # print 'fx:', sum_dx_i, 'fy:', sum_dy_i
        #
        # n_dx = len(dx_interp)
        #
        # ave_dx_i = sum_dx_i / n_dx
        # ave_dy_i = sum_dy_i / n_dx
        #
        # dx_i_deducted = dx_interp - ave_dx_i
        # dy_i_deducted = dy_interp - ave_dy_i
        #
        # center_x = 0
        # center_y = 0
        # sum_abs_dx = np.sum(np.abs(dx_interp))
        # sum_abs_dy = np.sum(np.abs(dy_interp))
        #
        # for i in range(0, len(dx_interp)):
        #     center_x += XXX[i] * np.abs(dx_interp[i]) / sum_abs_dx
        #     center_y += YYY[i] * np.abs(dy_interp[i]) / sum_abs_dy
        # print 'center:', center_x, center_y
        #
        #
        #
        # # center_x_quad = 0
        # # center_y_quad = 0
        # # sum_dx_quad = 0
        # # sum_dy_quad = 0
        # # for i in range(0, len(dx_interp)):
        # #     #     print np.exp(0.1*np.abs(dx_interp[i])) - 1
        # #     sum_dx_quad += np.exp(0.5 * np.abs(dx_interp[i])) - 1
        # #     sum_dy_quad += np.exp(0.5 * np.abs(dy_interp[i])) - 1
        # # print sum_dx_quad
        # # for i in range(0, len(dx_interp)):
        # #     center_x_quad += XXX[i] * (np.exp(0.5 * np.abs(dx_interp[i])) - 1) / sum_dx_quad
        # #     center_y_quad += YYY[i] * (np.exp(0.5 * np.abs(dy_interp[i])) - 1) / sum_dy_quad
        # #
        # # center_x = center_x_quad
        # # center_y = center_y_quad
        #
        # torque = 0
        # for i in range(0, len(dx_interp)):
        #     loc_diff_vec = np.array([XXX[i] - center_x, YYY[i] - center_y, 0])
        #     disp_vec = np.array([dx_i_deducted[i], dy_i_deducted[i], 0])
        #
        #     tau = np.cross(loc_diff_vec, disp_vec)
        #     #     print tau
        #     torque += tau[2]
        #
        # print 'torque: ', torque
        # # draw force and torque on image
        # # frame = np.ones((560, 440, 3))
        # cv2.arrowedLine(img_rgb, (int(np.round(center_x)), int(np.round(center_y))),
        #                 (int(np.round(center_x + 10* sum_dx_i/n_dx)) , int(np.round(center_y + 10*sum_dy_i/n_dx))),
        #                 (0, 255, 0), thickness=8, tipLength=0.3)
        # cv2.ellipse(img_rgb, (int(np.round(center_x)), int(np.round(center_y))), (50, 50),
        #             0, 0, torque*360/1000000, (0, 0, 255), thickness=10)
        #
        # df1 = df1.append({'Fx': sum_dx_i, 'Fy': sum_dy_i, 'Tz': torque/100.0}, ignore_index=True)
        #
        # cv2.imshow('arrow', img_rgb)

    count += 1

    print 'time elapsed: {}'.format(time.time() - start)
    print '=========================================================='

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
# cv2.imwrite('./tracking/image.jpg', img_rgb)

# df1.to_csv('./tracking/fxfyt.csv')

cap.release()
cv2.destroyAllWindows()
