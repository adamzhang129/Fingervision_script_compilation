
from undistort_crop_resize import *
import cv2
from sklearn.neighbors import KDTree
import numpy as np

import pandas as pd
pd.set_option('display.max_rows', 500)

from scipy import ndimage
from skimage import morphology, util, filters
import skimage
from scipy.interpolate import Rbf
import time
from pynhhd.nHHD import nHHD
from yaml import load


class Fv:
    def __init__(self, vid_index=0, interp_nx=30, interp_ny=30):
        self.cap = cv2.VideoCapture(vid_index)
        self.map1, self.map2 = undistort_setup()

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
            self.detector = cv2.SimpleBlobDetector(params)
        else:
            self.detector = cv2.SimpleBlobDetector_create(params)

        # params for interpolation and decomposition

        y_min = 20
        y_max = 480 - 20
        x_min = 120 - 80
        x_max = 520 + 80
        self.ROI = [y_min, y_max, x_min, x_max]
        self.width, self.height = x_max - x_min, y_max - y_min

        self.Nx, self.Ny = interp_nx, interp_ny
        x = np.linspace(0, self.width, self.Nx)
        y = np.linspace(0, self.height, self.Ny)

        self.X, self.Y = np.array(np.meshgrid(x, y))
        self.XX = self.X.ravel()  # flatten
        self.YY = self.Y.ravel()

        # decomposition object
        dx = float(self.width) / (self.Nx - 1)
        dy = float(self.height) / (self.Ny - 1)
        grid = (self.Nx, self.Ny)

        self.decomp_obj = nHHD(grid=grid, spacings=(dy, dx))

        self.loc_0 = []
        self.recent_loc = []
        self.count = 0
        self.vfield = []

        self.v_sum_mag = None
        self.v_sum_ang = None
        self.d_mag_sum = None
        self.sum_torque = None

    def crop(self, img):
        return img[self.ROI[0]:self.ROI[1], self.ROI[2]:self.ROI[3]]

    def blob_detect(self, img):
        keypoints = self.detector.detect(img)

        # print(len(keypoints))
        locs = []
        for i in range(0, len(keypoints)):
            locs.append([keypoints[i].pt[0], keypoints[i].pt[1]])

        # print(np.array(locs))
        return np.array(locs), keypoints


    def track(self):
        ret, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_ud = cv2.remap(gray, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        gray_crop = self.crop(gray_ud)

        loc, keypoints = self.blob_detect(gray_crop)

        if self.count == 0:

            self.loc_0 = loc.copy()
            self.recent_loc = loc.copy()
        elif self.count > 0:
            # print('============frame {}================'.format(count))
            # print(loc_0[1,:])
            kdt = KDTree(loc, leaf_size=30, metric='euclidean')
            dist, ind = kdt.query(self.recent_loc, k=1)
            thd = (dist < 14) * 1
            thd_nz = np.where(thd)[0]
            # update point if close enough point are detected
            self.recent_loc[thd_nz] = np.reshape(loc[ind[thd_nz]], (len(thd_nz), 2))

            # visualize the displacement field
            loc_v = 2 * self.recent_loc - self.loc_0  # diff vector


            # interpolation
            disp = self.recent_loc - self.loc_0

            dx, dy = disp[:, 0], disp[:, 1]
            x, y = self.recent_loc[:, 0], self.recent_loc[:, 1]

            interpolation_x = Rbf(x, y, dx)
            interpolation_y = Rbf(x, y, dy)

            dx_interp = interpolation_x(self.XX, self.YY)
            dy_interp = interpolation_y(self.XX, self.YY)
            mag = np.sqrt(dx_interp ** 2 + dy_interp ** 2)

            self.vfield = np.stack((dx_interp.reshape(self.Nx, self.Ny),
                                    dy_interp.reshape(self.Nx, self.Ny)), axis=2)
            # print self.vfield

            img_rgb = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2RGB)
            # draw image and save vectors
            for i in range(0, len(dx_interp)):
                cv2.arrowedLine(img_rgb, (int(np.around(self.XX[i])), int(np.around(self.YY[i]))),
                                (int(np.around(self.XX[i] + dx_interp[i])), int(np.around(self.YY[i] + dy_interp[i]))),
                                (0, 255, 255), thickness=2, tipLength=0.5)

            cv2.imshow('arrow_interp', img_rgb)

        self.count += 1

    def tracking_reset(self):
        self.recent_loc = []
        self.loc_0 = []
        self.count = 0

    def wrench_estimate(self):
        # ======calculate tangential force =========================
        # print self.vfield

        vx_sum = np.sum(self.vfield[:, :, 0])
        vy_sum = np.sum(self.vfield[:, :, 1])

        self.v_sum_mag = np.hypot(vx_sum, vy_sum)
        self.v_sum_ang = np.arctan2(vy_sum, vx_sum)

        # ======calculate normal  ===============
        self.decomp_obj.decompose(self.vfield, verbose=0)
        d = self.decomp_obj.d
        r = self.decomp_obj.r
        h = self.decomp_obj.h

        d = d.reshape(self.Nx * self.Ny, 2)
        r = r.reshape(self.Nx * self.Ny, 2)
        h = h.reshape(self.Nx * self.Ny, 2)

        mag_d = np.hypot(np.abs(d[:, 0]), np.abs(d[:, 1]))
        self.d_mag_sum = np.sum(mag_d, axis=0)
        # print 'total magnitude of d: {}'.format(sum_d)

        # ====== calculate torsional force =========================

        # calculate the index of maximum and minimum of potential field as rotation center
        R = self.decomp_obj.nRu
        R_max_ind = np.unravel_index(np.argmax(R), R.shape)
        R_max_loc = (self.X[R_max_ind], self.Y[R_max_ind])

        R_min_ind = np.unravel_index(np.argmin(R), R.shape)
        R_min_loc = (self.X[R_min_ind], self.Y[R_min_ind])

        # validate using first derivative
        i_max, j_max = R_max_ind
        i_min, j_min = R_min_ind
        # validate with finite difference symbols
        if i_max > 0 and i_max < self.Nx-1 and j_max > 0 and j_max < self.Ny-1:
            if (R[i_max, j_max] - R[i_max - 1, j_max]) > 0 and (R[i_max + 1, j_max] - R[i_max, j_max])< 0 \
                    and (R[i_max, j_max] - R[i_max, j_max - 1]) > 0 and (R[i_max, j_max + 1] - R[i_max, j_max]) < 0:
                    # this point is the local maxima
                    print 'max for R exist'
                    max_loc = R_max_loc
            else:
                max_loc = None
        else:
            max_loc = None

        if i_min > 0 and i_min < self.Nx-1 and j_min > 0 and j_min < self.Nx-1:
            if (R[i_min, j_min] - R[i_min - 1, j_min]) < 0 and (R[i_min + 1, j_min] - R[i_min, j_min]) > 0 \
                    and (R[i_min, j_min] - R[i_min, j_min - 1]) < 0 and (R[i_min, j_min + 1] - R[i_min, j_min]) > 0:
                    # this point is the local minima
                    print 'min for R exist'
                    min_loc = R_min_loc
            else:
                min_loc = None
        else:
            min_loc = None

        XY = np.stack((self.XX, self.YY), axis=1)
        torque_max = 0.0
        if max_loc != None:
            cor_diff = XY - R_max_loc
            torque = np.cross(cor_diff, r)
            torque_max = np.sum(torque, axis=0)

        torque_min = 0.0
        if min_loc != None:
            cor_diff = XY - R_min_loc
            torque = np.cross(cor_diff, r)
            torque_min = np.sum(torque, axis=0)

        self.sum_torque = (torque_max + torque_min) / (self.Nx * self.Ny)
        # print self.v_sum_mag, self.a_tan, self.b_tan
        self.F_tan = self.inv_linear_func(self.v_sum_mag, self.a_tan, self.b_tan)
        self.F_nor = self.inv_linear_func(self.d_mag_sum, self.a_nor, self.b_nor)
        self.F_tor = self.inv_linear_func(self.sum_torque, self.a_tor, self.b_tor)

        F_tan_x = self.F_tan*np.cos(self.v_sum_ang)
        F_tan_y = self.F_tan*np.sin(self.v_sum_ang)

        return F_tan_x, F_tan_y, self.F_nor, self.F_tor

    def load_yaml(self, path):
        params_path = file(path)

        params = load(params_path)

        self.a_tan = params['fittings']['tangential']['a']
        self.b_tan = params['fittings']['tangential']['b']
        self.a_nor = params['fittings']['normal']['a']
        self.b_nor = params['fittings']['normal']['b']
        self.a_tor = params['fittings']['torsional']['a']
        self.b_tor = params['fittings']['torsional']['b']

    def inv_linear_func(self, y, a, b):
        return (y - b) / a



if __name__ == "__main__":
    fv = Fv(0)
    fv.load_yaml('./fitting_param.yaml')
    count = 0
    while True:
        fv.track()
        if fv.count > 1:
            print fv.wrench_estimate()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fv.cap.release()
    cv2.destroyAllWindows()