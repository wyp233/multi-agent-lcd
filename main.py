from os import path, listdir
import numpy as np
import matplotlib.pyplot as plt
import initialization as ini
import algorithm as alg
import yaml

'''main code for agent A'''


class image:
    def __init__(self, descriptor, label, x_coordinate, z_coordinate):
        self.des = descriptor
        self.lab = label
        self.x = x_coordinate
        self.z = z_coordinate


A_SDS = []
B_SDS = []
LCD_A = []
LCD_B = []
LCD_AB = []

tp = 0
tn = 0
fp = 0
fn = 0

with open("lcd.yaml", 'r') as f:
    setting = yaml.safe_load(f)

A_mem_files = [path.join(setting['A_mem_path'], f) for f in listdir(setting['A_mem_path'])];
B_mem_files = [path.join(setting['B_mem_path'], f) for f in listdir(setting['B_mem_path'])];

ini.get_ground_truth_table(setting['pose_path'], setting['storage_path'], accuracy=3)
fl_B = 0

for fl in A_mem_files:  # agent A get 1 image than agent B get 1 image
    image.des, im_label_k = ini.get_descriptor(fl)
    x_coordinate, z_coordinate = ini.get_pose_info(setting['pose_path'], fl)
    tempA_image = image(image.des, im_label_k, x_coordinate, z_coordinate)
    # save the 1st flame to SDS
    if len(A_SDS) == 0:
        A_SDS.append(tempA_image)
        continue
    if alg.motion_check(tempA_image, A_SDS):
        alg.f_search(tempA_image, A_SDS, 1,
                     LCD_A)  # loop closure detection including coarse search and fine search......
    image.des, im_label_k = ini.get_descriptor(B_mem_files[fl_B])
    x_coordinate, z_coordinate = ini.get_pose_info(setting['pose_path'], B_mem_files[fl_B])
    tempB_image = image(image.des, im_label_k, x_coordinate, z_coordinate)
    if len(B_SDS) == 0:
        B_SDS.append(tempB_image)
        fl_B += 1
        continue
    if alg.motion_check(tempB_image, B_SDS):
        alg.f_search(tempB_image, B_SDS, 1, LCD_B)
        alg.f_search(tempB_image, A_SDS, 0, LCD_AB)
    fl_B += 1
alg.save_image(setting['storage_path'] + 'LCD A/', LCD_A)
alg.save_image(setting['storage_path'] + 'LCD B/', LCD_B)
alg.save_image(setting['storage_path'] + 'LCD AB/', LCD_AB)

# drawing the result
data1 = np.loadtxt(setting['pose_path'])

x_A = data1[setting['A_start_label']:setting['A_end_label'], 3]
z_A = data1[setting['A_start_label']:setting['A_end_label'], 11]

x_B = data1[setting['B_start_label']:setting['B_end_label'], 3]
z_B = data1[setting['B_start_label']:setting['B_end_label'], 11]

x_LCD = []
z_LCD = []

for k in range(len(LCD_AB)):
    x_LCD.append(data1[int(LCD_AB[k][0])][3])
    x_LCD.append(data1[int(LCD_AB[k][1])][3])
    z_LCD.append(data1[int(LCD_AB[k][0])][11])
    z_LCD.append(data1[int(LCD_AB[k][1])][11])

figure1 = plt.scatter(x_A, z_A, c='r', s=3)
plt.scatter(x_B, z_B, c='b', s=3)
plt.scatter(x_LCD, z_LCD, c='g', s=3)
# plt.plot(x_LCD, z_LCD, c='g')
plt.show()

ground_truth = ini.get_ground_truth_table(setting['pose_path'], setting['storage_path'], accuracy=3)
for i in LCD_AB:
    if ground_truth[int(i[0])][int(i[1])] == 1:
        tp += 1
    else:
        fp += 1
for i in LCD_A:
    if ground_truth[int(i[0])][int(i[1])] == 1:
        tp += 1
    else:
        fp += 1
for i in LCD_B:
    if ground_truth[int(i[0])][int(i[1])] == 1:
        tp += 1
    else:
        fp += 1
with open(setting['storage_path'] + 'confusion matrix.txt', 'w') as f:
    f.write('tp=' + str(tp) + ' ')
    f.write('fp=' + str(fp))
