from __future__ import print_function, division
import cPickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def compute_scaling_factor(im_shape,target_size,max_size):
    """
    :param im_shape: The shape of the image
    :param target_size: The min side is resized to the target_size
    :param max_size: The max side is kept less than max_size
    :return: The scale factor
    """
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    return im_scale


if __name__ == '__main__':
    data = cPickle.load(open('./cache/wider_train_train_gt_roidb.pkl', 'rb'))
    size = []
    for db in data:
        boxes = db['boxes']
        scale_factor = compute_scaling_factor(db['image_size'], 800, 1200)
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        w *= scale_factor
        h *= scale_factor
        s = np.maximum(w, h)
        size.append(s)
    size = np.hstack(size)
    num = len(size)
    print(num)
    sps = [0, 8, 16, 24, 32, 48, 64, 128, 256, 512, 4096]
    for i in range(1, len(sps)):
        print('[%d, %d)'%(sps[i-1], sps[i]), np.sum(np.logical_and(size >= sps[i-1], size < sps[i])) / num)
    plt.hist(size, range=(0, 256), bins=50)
    #plt.gca().set_xscale("log")
    plt.show()

