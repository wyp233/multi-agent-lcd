import numpy as np
import cv2
import math
import yaml

with open("lcd.yaml", "r") as f:
    setting = yaml.safe_load(f)


def l2_dist(d_query, d_train):
    return np.linalg.norm(d_query - d_train)


def exp_dist(d_query, d_train):
    return np.exp(
        -(np.linalg.norm(d_query - d_train, ord=2, axis=0, keepdims=False) / 2 * setting['para_exp_sigma'] *
          setting['para_exp_sigma']))


def cosine_dist(d_query, d_train):
    return np.squeeze(np.dot(d_query, d_train))


# Function switcher
def general_distance(d_query, d_train, ftype):
    """
    3 possible types:
                    'l2', 'exp', 'cosine'
    """
    if ftype == 'l2':
        return l2_dist(d_query, d_train)

    elif ftype == 'exp':
        return exp_dist(d_query, d_train)

    elif ftype == 'cosine':
        return cosine_dist(d_query, d_train)

    else:
        raise


def threshold_choice(type):
    """
    2 possible threshold types used in coarse and fine search:
                    'l2', 'exp',
    """
    if type == 'l2':
        return setting['para_l2_dis']

    if type == 'exp':
        return setting['para_exp_dis']

    else:
        raise


def geometric_consistency_check(path1, path2):
    """
    return 1 if top two frames pass the geometric consistency check
    """
    # load the image
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    # extract the keypoints
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # match the keypoints
    bf = cv2.BFMatcher()
    matches = bf.match(descriptors1, descriptors2)

    # delete outliner keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    if len(dst_pts) <= 4:  # the function needs at least 4 points to calculate the matrix
        return 0

    H, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 5.0) # 5 is used to judge whether this is a inliner
    # (considered as model distance)
    # H: transformation matrix
    # Compare the number of real keypoints and total keypoints
    match_num = np.count_nonzero(mask)
    if match_num / len(matches) > setting['para_geometric_check']:
        return 1
    else:
        return 0


def motion_check(n_image, SDS):  # compare with last saved keyframe in specific SDS
    if math.sqrt((SDS[-1].x - n_image.x) ** 2 + (SDS[-1].z - n_image.z) ** 2) > setting['para_sufficient_motion']:
        return True
    else:
        return False


def save_image(path, LCD):
    for i in range(len(LCD)):
        img1 = cv2.imread(setting['general_path'] + str(LCD[i][0]) + '.png')
        img2 = cv2.imread(setting['general_path'] + str(LCD[i][1]) + '.png')
        img3 = cv2.hconcat([img1, img2])
        cv2.imwrite(path + str(LCD[i][0]) + ' and ' + str(LCD[i][1]) + '.png', img3, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return 0


def f_search(image_d, SDS, SDS_switch, LCD):
    """

    :param image_d: the incoming of the image
    :param SDS: the datastructure of the agent
    :param SDS_switch: switch = 1 you can update the SDS, switch = 0 you can't update the SDS
    :return: 0 = no loop closure and don't update the SDS,
             1 = loop closure found or refuse to update the SDS
             2 = no loop closure, update the SDS according to the switch
    """
    # coarse search
    cos_sim = []
    N_neighbour_coarse = []
    SDS_update_marker = 0
    im_label_k = image_d.lab

    for i in range(len(SDS)):  # go through the SDS for k nearest neighbours
        # print(alg.general_distance(image_d.des, SDS[i].des, setting['fun_coarse_knn']))
        cos_sim.append(
            [general_distance(image_d.des, SDS[i].des, setting['fun_coarse_knn']), SDS[i].des, SDS[i].lab, 0])
        # [cos_sim, SDS keyframe data, keyframe label, inverse exponential kernel score or inverse L2 distance score ]

    if setting['fun_coarse_knn'] == 'cosine':
        N_neighbour_coarse = [x for x in
                              sorted(cos_sim, key=lambda x: x[0], reverse=True)[:setting['para_coarse_knn_k']]]
    elif setting['fun_coarse_knn'] == 'l2':
        N_neighbour_coarse = [x for x in
                              sorted(cos_sim, key=lambda x: x[0], reverse=False)[:setting['para_coarse_knn_k']]]
    # select K1 nearest neighbours

    for i in range(len(N_neighbour_coarse)):  # score calculation
        N_neighbour_coarse[i][3] = 1 / general_distance(image_d.des, N_neighbour_coarse[i][1],
                                                        setting['fun_coarse_score'])
        if N_neighbour_coarse[i][3] < threshold_choice(setting['fun_coarse_score']):
            N_neighbour_coarse[i] = []
    coa_candidate = list(filter(lambda x: x != [], N_neighbour_coarse))  # delete void data, prepare for fine search

    if len(coa_candidate) == 0:
        SDS_update_marker = 1  # no coarse candidate, LCD interrupted, query frame is considered as a new location

    # fine search
    N_neighbour_fine = []
    exp = []

    for i in range(len(coa_candidate)):
        np.squeeze(np.dot(image_d.des, coa_candidate[i][1]))
        exp.append(
            [general_distance(image_d.des, coa_candidate[i][1], setting['fun_fine_knn']), coa_candidate[i][1],
             coa_candidate[i][2], 0])
        # cos_sim, candidate data (descriptor), candidate label, exp score

    if setting['fun_fine_knn'] == 'cosine':
        N_neighbour_fine = [x for x in sorted(exp, key=lambda x: x[0], reverse=True)[:setting['para_fine_knn_k']]]
    elif setting['fun_fine_knn'] == 'l2':
        N_neighbour_fine = [x for x in sorted(exp, key=lambda x: x[0], reverse=False)[:setting['para_fine_knn_k']]]
        # select K2 nearest neighbours

    for i in range(len(N_neighbour_fine)):
        N_neighbour_fine[i][3] = general_distance(image_d.des, N_neighbour_fine[i][1],
                                                  setting['fun_fine_score'])  # exp kernel
        if N_neighbour_fine[i][3] < threshold_choice(setting['fun_fine_score']):
            N_neighbour_fine[i] = []  # delete void data
    fine_candidate = list(filter(lambda x: x != [], N_neighbour_fine))

    if len(fine_candidate) < 2:
        top_candidate = fine_candidate
    else:
        top_candidate = [x for x in
                         sorted(fine_candidate, key=lambda x: x[3], reverse=True)[:2]]  # select top 2 candidates

    if len(top_candidate) == 0:
        SDS_update_marker = 1  # LCD interrupted, query frame is considered as a new location

    # Estimate the decision quality
    if len(top_candidate) == 2:
        if top_candidate[0][3] / top_candidate[1][3] < setting['para_decision_quality']:
            # Consistent geometry check
            path1 = setting['general_path'] + str(im_label_k) + ".png"
            path2 = setting['general_path'] + str(top_candidate[0][2]) + ".png"
            if geometric_consistency_check(path1, path2) == 1:
                print(im_label_k, top_candidate[0][2])
                LCD.append([im_label_k, top_candidate[0][2]])
            else:
                SDS_update_marker = 1  # LCD interrupted, query frame is considered as a new location

    elif len(top_candidate) == 1:
        path1 = setting['general_path'] + str(im_label_k) + ".png"
        path2 = setting['general_path'] + str(top_candidate[0][2]) + ".png"
        if geometric_consistency_check(path1, path2) == 1:
            print(im_label_k, top_candidate[0][2])
            LCD.append([im_label_k, top_candidate[0][2]])
        else:
            SDS_update_marker = 1  # LCD interrupted, check whether the query frame can be considered as a new location

    if SDS_switch == 0:
        SDS_update_marker = 0
    if SDS_update_marker == 1:
        SDS.append(image_d)
        return 0  # no loop closure and  update the SDS
    else:
        return 1  # loop closure find or refuse to update the SDS
