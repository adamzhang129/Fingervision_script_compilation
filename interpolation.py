import cv2

import numpy as np

import pandas as pd

import os

pd.set_option('display.max_rows', 500)



img_dir = '../dataset1/images'
csv_dir = '../dataset1/csv'


img = cv2.imread(os.path.join(img_dir, '2000.jpg'))

vectors = pd.read_csv(os.path.join(csv_dir, '2000.csv'))

position = vectors[['x', 'y']].values
v_diff = vectors[['vx', 'vy']].values

position_t =  position + v_diff
# visualization
for i in range(0, len(position)):
    cv2.arrowedLine(img, (int(np.around(position[i, 0])), int(np.around(position[i, 1]))),
                                 (int(np.around(position_t[i, 0])), int(np.around(position_t[i, 1]))), (0, 0, 255), thickness=2)

cv2.imshow('show', img)
cv2.waitKey(0)
cv2.destroyAllWindows()