import argparse
import os
import pprint
import tensorflow as tf

from comrec_trainertester import ComRecTrainerTester
from model import VDSR
import matplotlib.pyplot as plt
import numpy as np
from imresize import *
from model_com import Com
from model_rec import Rec

if __name__ == '__main__':
# =======================================================
# [global variables]
# =======================================================
    pp = pprint.PrettyPrinter()
    args = None
    DATA_PATH = "./train/"
    TEST_DATA_PATH = "./data/test/"

# =======================================================
# [add parser]
# =======================================================
    parser = argparse.ArgumentParser()
    #===================== common configuration ============================================
    parser.add_argument("--exp_tag", type=str, default="ComRecCNN tensorflow. Implemented by Dohyun Kim")
    parser.add_argument("--gpu", type=int, default=3)  # -1 for CPU

    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=40)
    parser.add_argument("--stride_size", type=int, default=20)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--jpgqfactor", type= int, default =10)

    parser.add_argument("--train_subdir", default="BSD400")
    parser.add_argument("--test_subdir", default="Set5")
    parser.add_argument("--infer_imgpath", default="monarch.bmp")  # monarch.bmp
    parser.add_argument("--type", default="YCbCr", choices=["RGB","Gray","YCbCr"])#YCbCr type uses images preprocessesd by matlab
    parser.add_argument("--c_dim", type=int, default=3) # 3 for RGB, 1 for Y chaanel of YCbCr (but not implemented yet)
    parser.add_argument("--mode", default="train", choices=["train", "test", "inference", "test_plot"])

    #===================== configuration for compactor ======================================
    parser.add_argument("--base_lr_com", type=float, default=1e-2)
    parser.add_argument("--min_lr_com", type=float, default=1e-4)
    parser.add_argument("--lr_decay_rate_com", type=float, default=1e-2)
    parser.add_argument("--lr_step_size_com", type=int, default=50)  # 9999 for no decay
    parser.add_argument("--checkpoint_dir_com", default="checkpoint_com")
    parser.add_argument("--cpkt_itr_com", default=0)  # -1 for latest, set 0 for training from scratch
    parser.add_argument("--save_period_com", type=int, default=1)

    #===================== configuration for reconstructor ======================================
    parser.add_argument("--num_inner_rec",type = int, default = 18)
    parser.add_argument("--base_lr_rec", type=float, default=1e-2)
    parser.add_argument("--min_lr_rec", type=float, default=1e-4)
    parser.add_argument("--lr_decay_rate_rec", type=float, default=1e-2)
    parser.add_argument("--lr_step_size_rec", type=int, default=50)  # 9999 for no decay
    parser.add_argument("--checkpoint_dir_rec", default="checkpoint_rec")
    parser.add_argument("--cpkt_itr_rec", default=0)  # -1 for latest, set 0 for training from scratch
    parser.add_argument("--save_period_rec", type=int, default=1)



    parser.add_argument("--result_dir", default="result")
    parser.add_argument("--save_extension", default=".jpg", choices=["jpg", "png"])

    print("=====================================================================")
    args = parser.parse_args()
    if args.type == "YCbCr":
        args.c_dim = 1; #args.train_subdir += "_M"; args.test_subdir += "_M"
    elif args.type == "RGB":
        args.c_dim = 3;
    elif args.type == "Gray":
        args.c_dim = 1
    print("Eaxperiment tag : " + args.exp_tag)
    pp.pprint(args)
    print("=====================================================================")

# =======================================================
# [make directory]
# =======================================================
    if not os.path.exists(os.path.join(os.getcwd(), args.checkpoint_dir_com)):
        os.makedirs(os.path.join(os.getcwd(), args.checkpoint_dir_com))
    if not os.path.exists(os.path.join(os.getcwd(), args.checkpoint_dir_rec)):
        os.makedirs(os.path.join(os.getcwd(), args.checkpoint_dir_rec))
    if not os.path.exists(os.path.join(os.getcwd(), args.result_dir)):
        os.makedirs(os.path.join(os.getcwd(), args.result_dir))

# =======================================================
# [Main]
# =======================================================
    # -----------------------------------
    # system configuration
    # -----------------------------------
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = tf.ConfigProto()
    if args.gpu == -1: config.device_count = {'GPU': 0}
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # config.operation_timeout_in_ms=10000


    # -----------------------------------
    # build model
    # -----------------------------------
    with tf.Session(config = config) as sess:
        com = Com(sess = sess, args = args)
        rec = Rec(sess = sess, args = args, com = com)

        '''
        img = plt.imread("dataset/inference/monarch.bmp")
        result = com.inference(img)
        print(result.shape)
        result = rec.inference(result)
        print(result.shape)
        plt.imshow(result)
        plt.show()
        '''

        comrec_trainertester = ComRecTrainerTester(sess = sess, args = args, com = com, rec = rec)

        # -----------------------------------
        # train, test, inferecnce
        # -----------------------------------
        if args.mode == "train":
            comrec_trainertester.train()


        elif args.mode == "test":
            comrec_trainertester.test()


        elif args.mode == "inference":
            pass

        elif args.mode == "test_plot":
            pass



