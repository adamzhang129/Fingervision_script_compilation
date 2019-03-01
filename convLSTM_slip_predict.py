from convLSTM_slip_detection_1layer import ConvLSTMCell


import torch

channels = 3
hidden_size = 64

model = ConvLSTMCell(channels, hidden_size)

model.load_state_dict(torch.load('./saved_model/convlstm_model.pth'))

print model
if torch.cuda.is_available():
    # print 'sending model to GPU'
    model = model.cuda()


# ================ collect image frames ==========

from undistort_crop_resize import *

from sklearn.neighbors import KDTree

import cv2

import numpy as np
np.set_printoptions(threshold='nan')
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



def draw_interp_field(dx_interp, dy_interp, fx, fy):
    mag = np.sqrt(dx_interp ** 2 + dy_interp ** 2)

    dx_resized = (dx_interp.reshape(Nx, Ny) + 30) * 256 / 60
    dy_resized = (dy_interp.reshape(Nx, Ny) + 30) * 256 / 60
    mag_resized = (mag.reshape(Nx, Ny) + 30) * 256 / 60
    # print '====== ', dx_resized.max(), dx_resized.min()

    # fx, fy = 1, 1
    dx_large = cv2.resize(dx_resized, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    dy_large = cv2.resize(dy_resized, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    mag_large = cv2.resize(mag_resized, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

    # concated = np.concatenate((dx_large, dy_large, mag_large), axis=1)
    # concated = np.array(concated, dtype=np.uint8)
    dx_large = np.array(dx_large, dtype=np.uint8)
    dy_large = np.array(dy_large, dtype=np.uint8)
    mag_large = np.array(mag_large, dtype=np.uint8)

    dx_large = cv2.applyColorMap(dx_large, cv2.COLORMAP_JET)
    dy_large = cv2.applyColorMap(dy_large, cv2.COLORMAP_JET)
    mag_large = cv2.applyColorMap(mag_large, cv2.COLORMAP_JET)

    # concated = cv2.applyColorMap(concated, cv2.COLORMAP_JET)

    # cv2.imshow('dx dy mag', concated)

    cv2.imshow('dx', dx_large)
    cv2.imshow('dy', dy_large)
    cv2.imshow('mag', mag_large)




from scipy.interpolate import Rbf

count = 0

loc_0 = []
# interpolation grid width and length
Nx = 30
Ny = 30

N_frames = 10
frame_matrix = np.zeros((N_frames, 3, Nx, Ny))

frame_counter = 0


import time

while (True):
    start = time.time()
    # print count
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
    # cv2.imwrite('./tracking/image.jpg', gray_crop)

    loc, keypoints = blob_detect(gray_crop)
    # print(loc.shape)
    # print(loc)
    # im_with_keypoints = cv2.drawKeypoints(gray_crop, keypoints, np.array([]),
    #                                       (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow('keypoints', im_with_keypoints)
    # cv2.imwrite('./tracking/image2.jpg', im_with_keypoints)

    if count == 0:
        # print 'initiate tracking matrix'
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

        img_rgb = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB)
        # # draw image and save vectors
        for i in range(0, len(loc_0)):
            cv2.arrowedLine(img_rgb, (int(np.around(recent_loc[i, 0])), int(np.around(recent_loc[i, 1]))),
                            (int(np.around(loc_v[i, 0])), int(np.around(loc_v[i, 1]))), (0, 0, 255), thickness=2)
        # cv2.imshow('arrow', img_rgb)

        # df = pd.DataFrame(np.concatenate((recent_loc, loc_v), axis=1), columns=['x', 'y', 'xt', 'yt'])
        # df.to_csv('./tracking/vectors.csv')

        # =============  interpolation ====================
        disp = recent_loc - loc_0

        # if len(disp) > 179:
        #     disp = disp[:179, :]
        #     recent_loc = recent_loc[:179, :]

        dx, dy = disp[:, 0], disp[:, 1]
        x, y = recent_loc[:, 0], recent_loc[:, 1]

        interpolation_x = Rbf(x, y, dx)
        interpolation_y = Rbf(x, y, dy)

        xx = np.linspace(0, 560, Nx)
        yy = np.linspace(0, 440, Ny)

        XX, YY = np.array(np.meshgrid(xx, yy))

        XXX = XX.ravel()  # flatten
        YYY = YY.ravel()

        dx_interp = interpolation_x(XXX, YYY)
        dy_interp = interpolation_y(XXX, YYY)
        mag = np.sqrt(dx_interp ** 2 + dy_interp ** 2)

        img_rgb = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB)
        # draw image and save vectors
        for i in range(0, len(dx_interp)):
            cv2.arrowedLine(img_rgb, (int(np.around(XXX[i])), int(np.around(YYY[i]))),
                            (int(np.around(XXX[i] + dx_interp[i])), int(np.around(YYY[i] + dy_interp[i]))),
                            (0, 255, 255), thickness=2, tipLength=0.3)


        # print dx_interp.shape
        # draw_interp_field(dx_interp, dy_interp, fx=10, fy=10)

        # ======= store image frames
        dx_reshape = dx_interp.reshape(Nx, Ny)
        dy_reshape = dy_interp.reshape(Nx, Ny)
        mag_reshape = mag.reshape(Nx, Ny)

        # stack them along axis 0
        temp = [dx_reshape, dy_reshape, mag_reshape]
        matrix_3 = np.stack(temp, axis=0)

        frame_counter += 1
        if frame_counter <= N_frames:
            print '{} frames stored!'.format(frame_counter)


        frame_matrix[0:N_frames-1, :, :, :] = frame_matrix[1:N_frames, :, :, :]
        frame_matrix[N_frames-1, :, :, :] = matrix_3

        # print frame_matrix[:, 0,0,0]
        # print frame_matrix[0, :,:,:]
        frame_tensor = torch.from_numpy(frame_matrix)
        # print frame_tensor[:,0,0,0]

        # =============== do inference ===================
        x = frame_tensor.unsqueeze(1)
        # print x.shape

        if torch.cuda.is_available():
            # print 'sending input and target to GPU'
            x = x.type(torch.cuda.FloatTensor)
            # y = y.type(torch.cuda.FloatTensor)

        state = None
        out = None

        for t in range(0, N_frames):
            out, state = model(x[t], state)

        _, argmax = torch.max(out, 1)

        output = (argmax.data).cpu().numpy()

        if output == 1:
            # print 'slip occurring'
            cv2.putText(img_rgb, 'Slip occurring', (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        else:
            # print 'non-slip'
            cv2.putText(img_rgb, 'Nonslip', (10, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.imshow('with indication', img_rgb)

        # ======= concatenate all images =========================================================
        dx_resized = (dx_interp.reshape(Nx, Ny) + 30) * 256 / 60
        dy_resized = (dy_interp.reshape(Nx, Ny) + 30) * 256 / 60
        mag_resized = (mag.reshape(Nx, Ny) + 30) * 256 / 60
        # print '====== ', dx_resized.max(), dx_resized.min()

        fx = 6.2
        fy = 6.2

        dx_large = cv2.resize(dx_resized, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        dy_large = cv2.resize(dy_resized, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        mag_large = cv2.resize(mag_resized, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        # print dx_large.shape
        # concatenate 3 images to fit 560 (the size of captured image)
        black1 = np.ones((dx_large.shape[0], 1)) * (-256)
        black2 = np.ones((dx_large.shape[0], 1)) * (-256)
        concated = np.concatenate((dx_large, black1, dy_large, black2, mag_large), axis=1)
        concated = np.array(concated, dtype=np.uint8)

        # dx_large = np.array(dx_large, dtype=np.uint8)
        # dy_large = np.array(dy_large, dtype=np.uint8)
        # mag_large = np.array(mag_large, dtype=np.uint8)

        # dx_large = cv2.applyColorMap(dx_large, cv2.COLORMAP_JET)
        # dy_large = cv2.applyColorMap(dy_large, cv2.COLORMAP_JET)
        # mag_large = cv2.applyColorMap(mag_large, cv2.COLORMAP_JET)

        concated = cv2.applyColorMap(concated, cv2.COLORMAP_JET)

        # cv2.imshow('dx dy mag', concated)

        # concatenate with captured image
        images = np.concatenate((img_rgb, concated), axis=0)

        cv2.imshow('monitor window', images)


    count += 1

    # monitoring key input
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

    # print time.time() - start


cap.release()
cv2.destroyAllWindows()