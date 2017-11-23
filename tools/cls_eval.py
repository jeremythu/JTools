import cv2
import os,sys
import numpy as np


caffe_path = '/home/gaia/Code/caffe-test/python/'
if caffe_path not in sys.path:
    sys.path.insert(0, caffe_path)

import caffe


class CNNClassifier(object):
    def __init__(self, model_proto, model_weights, gpu_id, input_size,input_channel=3,mean_val=[128,128,128]):
        self._model_proto = model_proto
        self._model_weights = model_weights
        self._gpu_id = gpu_id
        self._input_size = input_size
        self._input_channel = input_channel
        self._mean_value = np.array(mean_val)

        if gpu_id == -1:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(gpu_id)

        self._net = caffe.Net(model_proto, model_weights, caffe.TEST)

    def image_preprocess(self, bgr_image):
        
        edge_removed_image = cut_black(bgr_image)
        edge_removed_image = cv2.resize(edge_removed_image, (self._input_size, self._input_size))
        edge_removed_image -= self._mean_value
        transposed_image = edge_removed_image.transpose(2,0,1)

        return transposed_image

    def net_forward(self, image, output_layer='prob'):
        image = image.reshape((1,)+image.shape)
        self._net.blobs['data'].reshape(*image.shape)
        self._net.blobs['data'].data[...] = image*1.0
        output = self._net.forward()

        #return self._net.blobs[output_layer].data
        return output[output_layer]

    def classify(self, image):
        transformed_image = self.image_preprocess(image)
        score_vec = self.net_forward(transformed_image)
        return score_vec


def cut_black(raw_image):
    col_forward_accumulation = []
    row_forward_accumulation = []
    col_backward_accumulation = []
    row_backward_accumulation = []

    gray_img = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)

    gray_img = gray_img * ( gray_img > 2)

    col_left = 0
    col_right = gray_img.shape[1]
    row_left = 0
    row_right = gray_img.shape[0]

    for i in range(gray_img.shape[0]):
        if sum(gray_img[i,:]) > gray_img.shape[0] * 1:
            row_left = i
            break

    for i in range(gray_img.shape[1]):
        if sum(gray_img[:, i]) >gray_img.shape[1] * 1:
            col_left = i
            break

    for i in xrange(-gray_img.shape[0], 0):
        if sum(gray_img[-i -1, : ]) >gray_img.shape[0] * 1:
            row_right = -i - 1
            break

    for i in xrange(-gray_img.shape[1], 0):
        if sum(gray_img[:, -i - 1]) > gray_img.shape[1] * 1:
            col_right = -i - 1
            break

    return np.asarray(raw_image[row_left: row_right, col_left:col_right],dtype=np.float32)


def top_k_from(all_res, k=5):
    sorted_inds = []
    for res in all_res:
        sorted_inds.append([res.argsort()[::-1][:k]])
    return sorted_inds


if __name__ == '__main__':
    
    single_test = 0
    image_list_test = 1

    #prototxt = '/home/gaia/Code/CAM/CAM/DR-Grading/deploy_inception_v3.prototxt'
    #model_weight = '/home/gaia/Code/CAM/CAM/DR-Grading/inception_v3_961_iter_40000.caffemodel'
    prototxt = '/home/gaia/Code/caffe/exp/noise_request/model/ResNet-50-deploy.prototxt'
    model_weight = '/home/gaia/Code/caffe/exp/noise_request/snapshot/_iter_10000.caffemodel'

    gpu_id = 1
    mean_val = [128,128,128]

    input_size = 224

    cnn_classifier = CNNClassifier(prototxt, model_weight, gpu_id, input_size)

    if single_test:
        image_file = '/home/gaia/Data/fundus/screen/label/003/IMAGES/IM008398.JPG'
        image = cv2.imread(image_file)
        
        score = cnn_classifier.classify(image)
        res = top_k_from(score, 1)[0][0][0]
        print res
    else:
        if image_list_test:
            image_list = '/home/gaia/Code/caffe/exp/noise_request/others-test.txt'

            counter = 0
            correct_num = 0 

            for line in open(image_list):
                image_file = line.strip().split(' ')[0]
                print 'process {}'.format(image_file)
                try:
                    image = cv2.imread(image_file)
                    score = cnn_classifier.classify(image)
                    res = top_k_from(score,1)[0][0][0]
                except:
                    print '{} error'.format(image_file)
                    continue
                
                counter += 1
                if str(res) == line.strip().split(' ')[1]:
                    correct_num += 1
            
            print 'result: {}/{}'.format(correct_num, counter)

        else:
            image_folder = '/home/share/data_tmp/res017/'
            saving_folder = '/home/share/data_tmp/res017-res/'

            if not os.path.exists(saving_folder):
                os.system('mkdir {}'.format(saving_folder))

            for image_file in os.listdir(image_folder):
                print 'process {}'.format(os.path.join(image_folder, image_file))
                image = cv2.imread(os.path.join(image_folder, image_file))
                score = cnn_classifier.classify(image)
                res = top_k_from(score,1)[0][0][0]
                print res
'''
            if res == 0:
                os.system('cp {} {}'.format(os.path.join(image_folder, image_file), os.path.join(saving_folder+'0', image_file)))
            elif res == 1:
                os.system('cp {} {}'.format(os.path.join(image_folder, image_file), os.path.join(saving_folder+'1', image_file)))
            elif res == 2:
                os.system('cp {} {}'.format(os.path.join(image_folder, image_file), os.path.join(saving_folder+'2', image_file)))
            elif res == 3:
                os.system('cp {} {}'.format(os.path.join(image_folder, image_file), os.path.join(saving_folder+'3', image_file)))
            elif res == 4: 
                os.system('cp {} {}'.format(os.path.join(image_folder, image_file), os.path.join(saving_folder+'4', image_file)))
'''             
