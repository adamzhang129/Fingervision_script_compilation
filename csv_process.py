import csv
import numpy as np
import pandas as pd

# with open('../wrench_displacement_processed.csv', 'w') as out:
#     # biased
#     reader = csv.reader(open('../wrench_displacement.csv', 'r'))
#     writer = csv.writer(out, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL) # 1
#
#     first_row = ['x', 'y', 'dx','dy','dtheta','fx_n','fy_n','fz_n', 'mx_n','my_n','mz_n','fx_d','fy_d','fz_d', 'mx_d','my_d','mz_d']
#     i = 0
#     for row in reader:
#         if i == 0:
#             writer.writerow(first_row) # 2
#
#         else:
#             # print row
#             row = np.array(map(float, row))
#             x, y = row[2:4]
#             dx,dy = row[2:4] - row[0:2]
#             dtheta = row[4]
#             fx_n, fy_n, fz_n, mx_n, my_n, mz_n = row[11:17] - row[5:11]
#             fx_d, fy_d, fz_d, mx_d, my_d, mz_d = row[17:23] - row[5:11]
#             new_row = [x, y, dx, dy, dtheta, fx_n, fy_n, fz_n,
#                            mx_n, my_n, mz_n, fx_d, fy_d, fz_d, mx_d, my_d, mz_d]
#             print(new_row)
#             writer.writerow(new_row)
#         i = i + 1
# print('====================================================================')
# with open('../loc_fxyz_mz.csv', 'w') as out:
#     reader = csv.reader(open('../wrench_displacement_processed.csv', 'r'))
#     writer = csv.writer(out, delimiter=',', quotechar="|", quoting=csv.QUOTE_MINIMAL) # 1
#
#     first_row = ['image_id','x/m', 'y/m', 'fx_d/N', 'fy_d/N', 'fz_d/N', 'mz_d/N*mm']
#     i = 0
#     for row in reader:
#         if i == 0:
#             writer.writerow(first_row) # 2
#
#         else:
#             # print row
#             row = np.array(map(float, row))
#             x, y = row[0:2]
#             fx_d, fy_d, fz_d = row[11:14]
#             mz_d = row[16]*1000
#
#             image_id = '%04d.jpg' % (i-1)
#
#             new_row = [image_id, x, y, fx_d, fy_d, fz_d, mz_d]
#             print(new_row)
#             writer.writerow(new_row)
#         i = i + 1
#
#
#
# first_row = ['image_id','x/m', 'y/m', 'fx_d/N', 'fy_d/N', 'fz_d/N', 'mz_d/N*mm']
#
# df = pd.read_csv('../loc_fxyz_mz.csv', skipinitialspace=True, usecols=first_row)
#
# for column in df[['x/m', 'y/m', 'fx_d/N', 'fy_d/N', 'fz_d/N', 'mz_d/N*mm']]:
#     # print(df[column])
#     mean = df[column].mean()
#     std = df[column].std()
#     df[column] = (df[column] - mean)/std
#
# df.to_csv('../loc_fxyz_mz_normalized.csv', columns=first_row, index=False)


# =========== process dataset2 wrench_loc.csv =========================
import yaml

# first_row = ['dtheta', 'loc_x', 'loc_y', 'fx', 'fy', 'fz', 'mx', 'my', 'mz']
#
# df1 = pd.read_csv('../dataset2/wrench_loc.csv', skipinitialspace=True, usecols=first_row)
# new_column = ['loc_x', 'loc_y', 'fx', 'fy', 'fz', 'mz']
#
# feature = df1[new_column].values
#
# df = pd.DataFrame(feature, columns=new_column)
#
#
# dict = {'Attribution': 'The statistic values of dataset2/wrench_loc.csv',
#         'loc_x': {'mean': None, 'std': None}, 'loc_y': {'mean': None, 'std': None} ,
#         'fx': {'mean': None, 'std': None} , 'fy': {'mean': None, 'std': None} ,
#         'fz': {'mean': None, 'std': None} , 'mz': {'mean': None, 'std': None} }
# for column in new_column:
#     # print(df[column])
#     if column == 'mz':
#         df[column] = df[column] * 1000  # change unit to N*mm
#     mean = df[column].mean()
#     std = df[column].std()
#     df[column] = (df[column] - mean) / std
#     print column + ' mean: ' + str(mean)
#     print ' stdev: ' + str(std)
#     # save to yaml file
#     dict[column]['mean'] = str(mean)
#     dict[column]['std'] = str(std)
#
# # print df
# df.to_csv('../dataset2/wrench_loc_normalized.csv', columns=new_column, index=False)
# with open('../dataset2/mean_std.yaml', 'w') as yl_file:
#     yaml.dump(dict, yl_file, default_flow_style=False)


first_row = ['dtheta', 'loc_x', 'loc_y', 'fx', 'fy', 'fz', 'mx', 'my', 'mz']
df = pd.read_csv('../dataset2/wrench_loc.csv', skipinitialspace=True, usecols=first_row)

dtheta = df['dtheta'].values
fx = df['fx'].values
fy = df['fy'].values


df1 = pd.DataFrame(columns=['fx', 'fy'])

for i in range(0, len(dtheta)):
    theta = 45*np.pi/180 + dtheta[i]

    rot = np.array([[np.sin(theta), -np.cos(theta)],[np.cos(theta), np.sin(theta)]])
    print 'rot: ', rot
    f_v = np.array([[fx[i]], [fy[i]]])
    print 'fv:', f_v
    fv_c = np.matmul(rot, f_v)
    print 'fx', fv_c[0,0], 'fy', fv_c[1, 0]

    df1 = df1.append({'fx': fv_c[0,0], 'fy': fv_c[1, 0]}, ignore_index=True)


df1.to_csv('../dataset2/wrench_loc_cam_coor.csv')