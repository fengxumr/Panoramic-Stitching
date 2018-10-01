# from __future__ import print_function  #
import cv2
import argparse
import os
import math
import numpy
import random

NFEATURES = 500
RANSACT_TIMES = NFEATURES * 3
RANSACT = 20
RATE_DIST = 0.8
RATE_PNTS = 0.3


def up_to_step_1(imgs, imgs_name, output_path="./output/step1"):
    try:
        os.stat(output_path)
    except:
        os.mkdir(output_path)
    imgs_all = []
    for i, img in enumerate(imgs):
        detector = cv2.xfeatures2d.SIFT_create(nfeatures=NFEATURES)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps = detector.detect(gray_img, None)
        kps_img = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        imgs_all.append[[os.path.join(output_path, 'feature_point_' + imgs_name[i]), kps_img]]
    for i in imgs_all:
        cv2.imwrite(i[0], i[1])


def get_kps_des(imgs):
    kps_list = []
    des_list = []
    detector = cv2.xfeatures2d.SIFT_create(NFEATURES)
    for i, img in enumerate(imgs):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, des = detector.detectAndCompute(gray_img, None)
        kps_list.append(kps)
        des_list.append(des)
    return kps_list, des_list

def knn_match(des_1, des_2):
    matches = []
    for k in range(len(des_1)):
        min_dist = math.inf
        min_2_dist = math.inf
        min_idx = -1
        for l in range(len(des_2)):
            dist = numpy.linalg.norm(des_1[k] - des_2[l])
            if dist < min_dist:
                min_2_dist = min_dist
                min_dist = dist
                min_idx = l
            elif dist < min_2_dist:
                min_2_dist = dist
            else:
                pass
        if min_dist / min_2_dist <= RATE_DIST:
            matches.append(cv2.DMatch(k, min_idx, 0, min_dist))        
    return matches

def up_to_step_2(imgs, imgs_name, output_path="./output/step2"):
    try:
        os.stat(output_path)
    except:
        os.mkdir(output_path)
    kps_list, des_list = get_kps_des(imgs)
    for j in range(len(imgs) - 1):
        matches = knn_match(des_list[j], des_list[j+1])
        if len(matches) >= NFEATURES * RATE_PNTS:
            match_img = cv2.drawMatches(imgs[j], kps_list[j], imgs[j+1], kps_list[j+1], matches, None, flags=2)
            cv2.imwrite(os.path.join(output_path, imgs_name[j].split('.')[0] + '_' + str(len(kps_list[j])) + '_' + imgs_name[j+1].split('.')[0] + '_' + str(len(kps_list[j])) + '.jpg'), match_img)


def find_trans_matrix(pts):
    A = []
    for i in range(len(pts)):
        x, y = round(pts[i][0][0]), round(pts[i][0][1])
        u, v = round(pts[i][1][0]), round(pts[i][1][1])
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = numpy.asarray(A)
    U, S, Vh = numpy.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    return L.reshape(3, 3)

def H_from_points(p1,p2):
    A = []
    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = numpy.asarray(A)
    U, S, Vh = numpy.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    H = L.reshape(3, 3)
    return H

def diff_after_trans(pts, mtx):
    p_1_cul_data = numpy.dot(mtx, numpy.append(numpy.array(pts[0]), 1).T)
    p_1_cal = p_1_cul_data[:2] / p_1_cul_data[2]
    return math.sqrt((pts[1][0] - p_1_cal[0]) ** 2 + (pts[1][1] - p_1_cal[1]) ** 2)

def find_homography(pairs):
    max_cnt = 0
    max_mtx = []
    for i in range(RANSACT_TIMES):
        pts_Idx = random.sample(range(len(pairs)), 4)
        mtx = find_trans_matrix([pairs[x] for x in pts_Idx])
        mtx_jy = H_from_points([pairs[x][0] for x in pts_Idx], [pairs[x][1] for x in pts_Idx])
        cnt = 0
        for p in pairs:
            if diff_after_trans(p, mtx) <= RANSACT:
                cnt += 1
        if cnt > max_cnt:
            max_cnt = cnt
            max_mtx = mtx
    good_pairs = []
    for p in pairs:
        if diff_after_trans(p, max_mtx) <= RANSACT:
            good_pairs.append(p)
    min_diff_sum = math.inf
    best_mtx = []
    for j in range(RANSACT_TIMES):
        diff_sum = 0
        pts_Idx = random.sample(range(len(good_pairs)), 4)
        mtx = find_trans_matrix([pairs[x] for x in pts_Idx])
        for p in good_pairs:
            diff_sum += diff_after_trans(p, mtx)
        if diff_sum < min_diff_sum:
            min_diff_sum = diff_sum
            best_mtx = mtx
    return best_mtx

def warp_perspective(img, mtx):
    co = []
    a_max = -math.inf
    a_min = math.inf
    b_max = -math.inf
    b_min = math.inf
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            c = mtx[2][0] * j + mtx[2][1] * i + mtx[2][2]
            a = int(round((mtx[0][0] * j + mtx[0][1] * i + mtx[0][2]) / c))
            b = int(round((mtx[1][0] * j + mtx[1][1] * i + mtx[1][2]) / c))
            co.append([b, a, i, j])
            if a > a_max:
                a_max = a
            if a < a_min:
                a_min = a
            if b > b_max:
                b_max = b
            if b < b_min:
                b_min = b
    warp_img = numpy.zeros([b_max - b_min + 1, a_max - a_min + 1, 3], dtype=numpy.uint8)
    for k in co:
        warp_img[k[0] - b_min][k[1] - a_min] = img[k[2]][k[3]]
        b_extra = k[0] - b_min + 1 if k[0] - b_min + 1 <= b_max - b_min else b_max - b_min
        warp_img[b_extra][k[1] - a_min] = img[k[2]][k[3]]
        a_extra = k[1] - a_min + 1 if k[1] - a_min + 1 <= a_max - a_min else a_max - a_min
        warp_img[k[0] - b_min][a_extra] = img[k[2]][k[3]]
        warp_img[b_extra][a_extra] = img[k[2]][k[3]]
    return warp_img


def up_to_step_3(imgs, imgs_name, output_path="./output/step3"):
    try:
        os.stat(output_path)
    except:
        os.mkdir(output_path)
    kps_list, des_list = get_kps_des(imgs)
    for j in range(len(imgs) - 1):
        matches = knn_match(des_list[j], des_list[j+1])
        if len(matches) >= NFEATURES * RATE_PNTS:
            pairs_list = [[kps_list[j][m.queryIdx].pt, kps_list[j+1][m.trainIdx].pt] for m in matches]
            homo_matrix_1 = find_homography(pairs_list)
            img_1 = warp_perspective(imgs[j], homo_matrix_1)
            cv2.imwrite(os.path.join(output_path, imgs_name[j].split('.')[0] + '_to_' + imgs_name[j+1].split('.')[0] + '.jpg'), img_1)
            homo_matrix_2 = numpy.linalg.inv(homo_matrix_1)
            img_2 = warp_perspective(imgs[j+1], homo_matrix_2)
            cv2.imwrite(os.path.join(output_path, imgs_name[j+1].split('.')[0] + '_to_' + imgs_name[j].split('.')[0] + '.jpg'), img_2)


def up_to_step_4(imgs, warp_imgs, output_path="./output/step4"):
    try:
        os.stat(output_path)
    except:
        os.mkdir(output_path)
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step",
        help="compute image stitching pipeline up to this step",
        type=int
    )

    parser.add_argument(
        "input",
        help="a folder to read in the input images",
        type=str
    )

    parser.add_argument(
        "output",
        help="a folder to save the outputs",
        type=str,
    )

    args = parser.parse_args()

    imgs_name = []
    imgs = []
    for filename in sorted(os.listdir(args.input)):
        imgs_name.append(filename)
        img = cv2.imread(os.path.join(args.input, filename))
        imgs.append(img)


    if args.step == 1:
        print("Running step 1")
        up_to_step_1(imgs, imgs_name, args.output)
    elif args.step == 2:
        print("Running step 1")
        up_to_step_1(imgs, imgs_name)
        print("Running step 2")
        up_to_step_2(imgs, imgs_name, args.output)
    elif args.step == 3:
        print("Running step 1")
        up_to_step_1(imgs, imgs_name)
        print("Running step 2")
        up_to_step_2(imgs, imgs_name)
        print("Running step 3")
        up_to_step_3(imgs, imgs_name, args.output)
    elif args.step == 4:
        print("Running step 1")
        up_to_step_1(imgs, imgs_name)
        print("Running step 2")
        up_to_step_2(imgs, imgs_name)
        print("Running step 3")
        warp_imgs = up_to_step_3(imgs, imgs_name)
        print("Running step 4")
        up_to_step_4(imgs, warp_imgs, args.output)
