"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import matplotlib.pyplot as plt
import scipy.misc
import scipy.ndimage
import scipy.io
import math
import numpy as np
import sys
from imresize import *
import glob, os, re
import cv2
from tqdm import tqdm
import tensorflow as tf

def rgb2ycbcr(im):
    '''
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return np.uint8(ycbcr)
    '''
    return cv2.cvtColor(im,cv2.COLOR_RGB2YCR_CB)[:,:,[0,2,1]]


def ycbcr2rgb(im):
    '''
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)
    '''


    temp =  cv2.cvtColor(im[:,:,[0,2,1]], cv2.COLOR_YCR_CB2RGB)
    return temp

def augumentation(img_sequence):
    augmented_sequence = []
    for img in img_sequence:
        for _ in range(3):
            rot_img = np.rot90(img)
            augmented_sequence.append(rot_img)

        flipped_img = np.fliplr(img)
        for _ in range(4):
            rot_flipped_img = np.rot90(flipped_img)
            augmented_sequence.append(rot_flipped_img)

    img_sequence.extend(augmented_sequence)
    return img_sequence


def imsave(image, path):
    # image = image - np.min(image)
    # image = image / np.max(image)
    # image = np.clip(image, 0, 1)
    # return plt.imsave(path, image)
    return scipy.misc.imsave(path, image)  # why different?







# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# inpur_setupt_eval. Using Matlab file (.m) upscaled with matlab bicubic
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def input_setup_eval(args, mode):
# ===========================================================
# [input setup] / split image
# ===========================================================
    sub_input_sequence = []
    sub_label_sequence = []

# ----------------------------------------------------------------
# [input setup] / split image - for trainset and testset
# ----------------------------------------------------------------
    if mode == "train" :
        inputs_, labels_= get_image("train/" + args.train_subdir, args = args)
        for (input_,label_) in zip(inputs_,labels_):
            h, w, _ = input_.shape  # only for R,G,B image

            for x in range(0, h - args.patch_size + 1, args.patch_size):
                for y in range(0, w - args.patch_size + 1, args.patch_size):
                    sub_input = input_[x:x + args.patch_size, y:y + args.patch_size, :]
                    sub_label = label_[x:x + args.patch_size, y:y + args.patch_size, :]
                    sub_input_sequence.append(sub_input)
                    sub_label_sequence.append(sub_label)

        return sub_input_sequence, sub_label_sequence

    elif mode == "test":
        nxy = []
        sub_input_sequence, sub_label_sequence = get_image("test/"+args.test_subdir, args=args)
        return sub_input_sequence, sub_label_sequence


def get_image(data_path, args):
    scale = args.scale
    l = glob.glob(os.path.join(data_path, "*"))
    l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
    img_list = []
    for f in l:
        if os.path.exists(f):
            if os.path.exists(f[:-4] + "_2.mat"): img_list.append([f, f[:-4] + "_2.mat", 2])
            if os.path.exists(f[:-4] + "_3.mat"): img_list.append([f, f[:-4] + "_3.mat", 3])
            if os.path.exists(f[:-4] + "_4.mat"): img_list.append([f, f[:-4] + "_4.mat", 4])

    input_list = []
    gt_list = []
    for pair in img_list:

        mat_dict = scipy.io.loadmat(pair[1])
        input_img = None
        if "img_" + str(scale) in mat_dict:
            input_img = mat_dict["img_" + str(scale)]
        else:
            continue
        if (args.c_dim == 3):
            input_img = np.stack([input_img, input_img, input_img], axis=-1)
        else:
            input_img = np.expand_dims(input_img, axis=-1)
        input_list.append(input_img)

        gt_img = scipy.io.loadmat(pair[0])['img_raw']
        if (args.c_dim == 3):
            gt_img = np.stack([gt_img, gt_img, gt_img], axis=-1)
        else:
            gt_img = gt_img.reshape(gt_img.shape[0], gt_img.shape[1], 1)  # np.expand_dims(gt_img,axis=-1)
        gt_list.append(gt_img)
    return input_list, gt_list
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%












# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# input_setup for demo. For RGB
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def input_setup_demo(args, mode):
    # ===========================================================
    # [input setup] / split image
    # ===========================================================
    sub_input_sequence = []
    sub_label_sequence = []

    # ----------------------------------------------------------------
    # [input setup] / split image - for trainset and testset
    # ----------------------------------------------------------------
    if mode == "train":
        data = prepare_data( args=args, mode=mode)
        for i in tqdm(range(len(data))):
            input_, label_ = preprocess(data[i],args, centercrop=True)  # normalized full-size image
            h, w, _ = input_.shape

            for x in range(0, h - args.patch_size + 1, args.stride_size):
                for y in range(0, w - args.patch_size + 1, args.stride_size):
                    sub_input = input_[x:x + args.patch_size, y:y + args.patch_size, :]
                    sub_label = label_[x:x + args.patch_size, y:y + args.patch_size, :]

                    sub_input_sequence.append(sub_input)
                    sub_label_sequence.append(sub_label)

        return sub_input_sequence, sub_label_sequence

    elif mode == "test":
        data = prepare_data(args=args, mode=mode)
        for i in range(len(data)):
            input_, label_ = preprocess(data[i], args)  # normalized full-size image
            sub_input_sequence.append(input_)
            sub_label_sequence.append(label_)
        return sub_input_sequence, sub_label_sequence





def prepare_data(args, mode):
    if mode == "train":
        data_dir = os.path.join(os.getcwd(),"dataset", mode, args.train_subdir)
        data = glob.glob(os.path.join(data_dir, "*"))

    elif mode == "test":
        data_dir = os.path.join(os.getcwd(), "dataset",mode, args.test_subdir)
        data = glob.glob(os.path.join(data_dir, "*"))

    return data






def preprocess(path, args, centercrop = False):
    image = plt.imread(path)
    if len(image.shape) < 3:
        image = np.stack([image] * 3, axis=-1)
    image = rgb2ycbcr(image)
    if np.max(image) > 1: image = (image / 255).astype(np.float32)


    if centercrop : image_croped = image[image.shape[0]//2-90:image.shape[0]//2+90,image.shape[1]//2-90:image.shape[1]//2+90]
    else : image_croped = modcrop(image, args.scale)


    if args.mode == "train" or args.mode == "test":
        label_ = image_croped
        input_ = label_

        #input_ = cv2.resize(image_croped,None,fx=1 / args.scale, fy=1 / args.scale, interpolation = cv2.INTER_AREA)
        #input_ = cv2.resize(input_,None,fx=args.scale, fy=args.scale, interpolation = cv2.INTER_CUBIC)
        return input_ ,label_







def modcrop(image, scale=3):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






