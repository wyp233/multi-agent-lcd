import numpy as np
import cv2
import re
from os import path
import onnxruntime as ort
import math


def get_ground_truth_table(data_path, storage_path, accuracy):  # accuracy usually = 3
    data = np.loadtxt(data_path)
    figure = []
    ground_truth_table = [[0 for i in range(len(data))] for i in range(len(data))]
    for i in range(len(data)):  # loading the data
        figure.append([i, data[i, 3], data[i, 11]])
    for k in range(len(data)):
        for i in range(len(data)):
            if math.sqrt((figure[k][1] - figure[i][1]) ** 2 + (figure[k][2] - figure[i][2]) ** 2) <= accuracy \
                    and abs(figure[k][0] - figure[i][0]) > 50:
                ground_truth_table[i][k] = 1
            else:
                ground_truth_table[i][k] = 0
    np.savetxt(storage_path + 'ground truth table.txt', ground_truth_table, fmt='%d')
    return ground_truth_table


def get_descriptor(fl):
    """

    :param fl: the direct path of the image
    :return: the descriptor of the image and the image label

    """
    sz = (160, 120)  # Required width, height for CALC

    model_fname = "D:/wyp/stage 3 final project/models onnx/model.onnx"

    providers = ['CPUExecutionProvider']
    ort.set_default_logger_severity(3)
    ort_sess = ort.InferenceSession(model_fname, providers=providers)

    im = cv2.imread(fl)
    im_label_k = re.match('.*?([0-9]+)$', path.splitext(path.basename(fl))[0]).group(1)

    img_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    if im.shape[2] > 1:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, sz, interpolation=cv2.INTER_CUBIC)
    im = np.float32(im) * 1.0 / 255.0
    im = np.expand_dims(im, axis=(0, 3))

    d_onnx = np.squeeze(ort_sess.run(None, {'X1_orig': im}))
    d_onnx /= np.linalg.norm(d_onnx)

    return d_onnx, im_label_k


def get_pose_info(pose_path, fl):
    data = np.loadtxt(pose_path)
    im_label = re.match('.*?([0-9]+)$', path.splitext(path.basename(fl))[0]).group(1)
    x_coordinate = data[int(im_label)][3]
    z_coordinate = data[int(im_label)][11]
    return x_coordinate, z_coordinate
