import cv2
import xml.etreeElementTree as ET
import numpy as np


def read_box_axis_from_xml(xml_file):
    tree = ET.parse(xml_file)
    objs = tree.findall('object')

    num_objs = len(objs)

    boxes = np.zeros((nnum_objs, 4), dtype=np.uint16)

    for idx, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        x_upperleft = float(bbox.find('xmin').text)-1
        y_upperleft = float(bbox.find('ymin').text)-1
        x_lowerright = float(bbox.find('xmax').text)-1
        y_lowerright = float(bbox.find('ymax').text)-1

        boxes[idx,:] = [x_upperleft, y_upperleft, x_lowerright, y_lowerright]

    return boxes
