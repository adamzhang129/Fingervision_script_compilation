import numpy as np
import pandas as pd

import os

import matplotlib  as mpl
import matplotlib.pyplot as plt

import cv2


mpl.rc('figure',  figsize=(22*1.5, 12*1.5))
mpl.rc('image', cmap='gray')


csv_path = '../dataset2/csv'
img_path = '../dataset2/images'

from scipy.interpolate import Rbf
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable



img_interp_path = '../dataset2/interpolation/images'
contour_dx_path = '../dataset2/interpolation/contour/dx'
contour_dy_path = '../dataset2/interpolation/contour/dy'
contour_mag_path = '../dataset2/interpolation/contour/mag'

vec_interp_path = '../dataset2/interpolation/vectors'

fig1 = plt.figure(1, figsize=(16, 16))
ax1 = fig1.add_subplot(111)
fig2 = plt.figure(2, figsize=(16, 16))
ax2 = fig2.add_subplot(111)
fig3 = plt.figure(3, figsize=(16, 16))
ax3 = fig3.add_subplot(111)
count = 0

df1 = pd.DataFrame(columns=['center_x', 'center_y', 'fx', 'fy', 'torque'])
for filename in os.listdir(csv_path):
    print 'num {} image '.format(count)

    ind = filename.split('.')[0]
    image_name = ind + '.jpg'
    image_file = os.path.join(img_path, image_name)
    csv_file = os.path.join(csv_path, filename)
    #     print image_path
    # read in the csv and image
    df = pd.read_csv(csv_file)
    # img = plt.imread(image_file)
    #
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # draw arrows on images
    #     print df
    x = df['x'].values
    y = df['y'].values
    xt = df['xt'].values
    yt = df['yt'].values


    if len(x) > 179:
        x = x[:179]
        y = y[:179]
        xt = xt[:179]
        yt = yt[:179]

    # interpolation

    # =================================================
    # interpolate denser grid data in x and y separately
    dx = xt - x
    dy = yt - y

    #     print dx.shape, dy.shape

    interpolation_x = Rbf(x, y, dx)
    interpolation_y = Rbf(x, y, dy)

    Nx = 30
    Ny = 30
    xx = np.linspace(0, 560, Nx)
    yy = np.linspace(0, 440, Ny)

    XX, YY = np.array(np.meshgrid(xx, yy))

    XXX = XX.ravel()  # flatten
    YYY = YY.ravel()

    dx_interp = interpolation_x(XXX, YYY)
    dy_interp = interpolation_y(XXX, YYY)

    # save interpolated vectors
    feature = np.concatenate((XXX.reshape(len(XXX), 1), YYY.reshape(len(XXX), 1),
                              dx_interp.reshape(len(XXX), 1), dy_interp.reshape(len(XXX), 1)), axis=1)
    Df = pd.DataFrame(feature, columns=['x', 'y', 'dx', 'dy'])
    # Df.to_csv(os.path.join(vec_interp_path, filename), index=False)

    #     print XXX.min(), XXX.max(), YYY.min(), YYY.max()

    mag = np.sqrt(dx_interp ** 2 + dy_interp ** 2)

    #     print dx_interp.shape
    #
    # print 'processing image: ' + image_name
    # for i in range(0, len(mag)):
    #     cv2.arrowedLine(img_rgb, (int(np.around(XXX[i])), int(np.around(YYY[i]))),
    #                     (int(np.around(XXX[i] + dx_interp[i])), int(np.around(YYY[i] + dy_interp[i]))),
    #                     (0, 255, 255), thickness=2, tipLength=0.3)
    #
    # # save the image with interp vectors
    # #     print os.path.join(img_interp_path, image_name)
    #
    # img_rgb = cv2.flip(img_rgb, 0)
    # cv2.imwrite(os.path.join(img_interp_path, image_name), img_rgb)

    # calculate center of force and torque and fx and fy
    sum_dx_i = np.sum(dx_interp)
    sum_dy_i = np.sum(dy_interp)

    n_dx = len(dx_interp)

    ave_dx_i = sum_dx_i / n_dx
    ave_dy_i = sum_dy_i / n_dx

    dx_i_deducted = dx_interp - ave_dx_i
    dy_i_deducted = dy_interp - ave_dy_i

    center_x = 0
    center_y = 0
    sum_abs_dx = np.sum(np.abs(dx_interp))
    sum_abs_dy = np.sum(np.abs(dy_interp))

    for i in range(0, len(dx_interp)):
        center_x += XXX[i] * np.abs(dx_interp[i]) / sum_abs_dx
        center_y += YYY[i] * np.abs(dy_interp[i]) / sum_abs_dy
    print 'center:', center_x, center_y

    torque = 0
    for i in range(0, len(dx_interp)):
        loc_diff_vec = np.array([XXX[i] - center_x, YYY[i] - center_y, 0])
        disp_vec = np.array([dx_i_deducted[i], dy_i_deducted[i], 0])

        tau = np.cross(loc_diff_vec, disp_vec)
        #     print tau
        torque += tau[2]

    # print 'torque: ', torque
    df1 = df1.append({'center_x': center_x, 'center_y': center_y, 'fx': sum_dx_i, 'fy':sum_dy_i, 'torque': torque},
                   ignore_index=True)












    # ax = fig.add_subplot(2, 2, 1)
    # plt.imshow(img_rgb)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # ============== save interpolated contour ==========================

    # drawing contour

    # plt.figure(1) # make the figure active again
    # ax1.set_aspect('equal')
    # cs = ax1.contourf(xx, yy, dx_interp.reshape(Nx, Ny), 25, cmap=plt.cm.coolwarm,
    #                  vmax=abs(dx_interp).max(), vmin=dx_interp.min())
    # plt.xlabel('X/pixel', fontsize=15)
    # plt.ylabel('Y/pixel', fontsize=15)
    # # create an axes on the right side of ax. The width of cax will be 5%
    # # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # # draw colorbar
    # plt.colorbar(cs, cax=cax)
    # plt.savefig(os.path.join(contour_dx_path, image_name), bbox_inches='tight')
    #
    # #     print xx.shape, yy.shape, dy_interp.reshape(Nx,Ny).shape
    #
    # plt.figure(2)
    # ax2.set_aspect('equal')
    # cs = ax2.contourf(xx, yy, dy_interp.reshape(Nx, Ny), 25, cmap=plt.cm.coolwarm,
    #                  vmax=abs(dy_interp).max(), vmin=dy_interp.min())
    # plt.xlabel('X/pixel', fontsize=15)
    # plt.ylabel('Y/pixel', fontsize=15)
    # # create an axes on the right side of ax. The width of cax will be 5%
    # # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    # divider = make_axes_locatable(ax2)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # # draw colorbar
    # plt.colorbar(cs, cax=cax)
    # plt.savefig(os.path.join(contour_dy_path, image_name), bbox_inches='tight')
    #
    # plt.figure(3)
    # ax3.set_aspect('equal')
    # cs = ax3.contourf(xx, yy, mag.reshape(Nx, Ny), 25, cmap=plt.cm.coolwarm,
    #                  vmax=abs(mag).max(), vmin=mag.min())
    # plt.xlabel('X/pixel', fontsize=15)
    # plt.ylabel('Y/pixel', fontsize=15)
    # # create an axes on the right side of ax. The width of cax will be 5%
    # # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    # divider = make_axes_locatable(ax3)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # # draw colorbar
    # plt.colorbar(cs, cax=cax)
    #
    # plt.savefig(os.path.join(contour_mag_path, image_name), bbox_inches='tight')

    #     plt.subplot(5, 2, count+1)
    #     plt.imshow(img_rgb, origin='lower')
    #     plt.xlabel('X/pixel')
    #     plt.ylabel('Y/pixel')

    # plt.draw()
    # plt.pause(0.1)


    count += 1

df1.to_csv('../dataset2/estimate_model_setby_manual.csv')