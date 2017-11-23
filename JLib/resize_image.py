import cv2
import numpy as np


def center_crop(image):
    """
    extract the square image from rectangle src image. The size of the square equals to the short side.
    :param image: src rectangle image
    :return: square image
    """

    if image.shape[0] == image.shape[1]:
        return image
    elif image.shape[0] < image.shape[1]:
        offset = (image.shape[1]-image.shape[0])/2
        return image[:, offset:offset+image.shape[0]]
    else:
        offset = (image.shape[0]-image.shape[1])/2
        return image[offset:offset+image.shape[1], : ]

def ratio_protected_resize(image, base_size):
    """
    resize the image with constant size ratio
    :param image: src image
    :param base_size: The size the shorter size being resized to
    :return: resized image
    """
    if image.shape[0] == image.shape[1]:
        return cv2.resize(image, (base_size, base_size))
    elif image.shape[0] < image.shape[1]:
        return cv2.resize(image, (image.shape[1]*base_size/image.shape[0], base_size))
    else:
        return cv2.resize(image, (base_size, image.shape[1]*base_size/image.shape[0]))

def center_resize(image, base_size):
    """
    resizing an image with protected size ratio and crop the central square image
    :param image: src image
    :param base_size: square size
    :return square image
    """
    resized_image = ratio_protected_resize(image, base_size)
    return center_crop(resized_image)

def _generate_grid_axis(end_t, grid_size, overlap_size):
    start_pos = 0
    end_pos = grid_size

    start = [start_pos]
    end = [end_pos]

    while end_pos < end_t:
        start_pos = end_pos - overlap_size
        end_pos = start_pos + grid_size
        start.append(start_pos)
        end.append(end_pos)

    start = start[:-1]
    end = end[:-1]
    start.append(end_t-1-grid_size)
    end.append(end_t-1)

    return start, end

def grid_image(image, grid_size=512, overlap=0.15):
    overlap_size = int(grid_size*overlap)

    start_y, end_y = _generate_grid_axis(image.shape[0], grid_size, overlap_size)
    start_x, end_x = _generate_grid_axis(image.shape[1], grid_size, overlap_size)
    return start_x, end_x, start_y, end_y
