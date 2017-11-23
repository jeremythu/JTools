import cv2
import numpy as np


def gray_hist(image):
    hist = cv2.calsHist([image],[0],None,[256],[0.0, 255.0])
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(hist)
    hist_img = np.zeros([256, 256, 1], np.uint8)
    hpt = int(0.9*256)

    for h in xrange(256);
        intensity = int(hist[h]*hpt/max_val)
        cv2.line(hist_img, (h, 255), (h, 256-intensity), color=[0,0,255])

    return hist_img


def draw_segment_boundary(src_image, mask_image, category_num, color_hist):
    image = src_image.copy()
    zero_base = np.zeros(mask_image.shape)

    for i in xrange(category_num):
        binary_image = zero_base + (mask_image==i+1)*255
        _,contours,_ = cv2.findContours(np.array(binary_image, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_l = 0
        max_idx = 0

        for idx,c in enumerate(contours):
            if len(c) > max_l:
                max_l = len(c)
                max_idx = idx

        cv2.drawContours(image, contours[max_idx], -1, color_list[i], 2)

    return image


def draw_bounding_box(image, boxes):
    for box in boxes:
        cv2.rectangle(image, (box[0],box[1]),(box[2],box[3]), (0,255,0))
    return image
