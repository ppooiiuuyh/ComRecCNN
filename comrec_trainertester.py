import tensorflow as tf
import numpy as np

from model_com import Com
from model_rec import Rec
from utils import *
from metrics import *

import time



class ComRecTrainerTester():
# ==========================================================
# init
# ==========================================================
    def __init__(self,sess, args, com, rec):
        self.args = args
        self.sess = sess
        self.com = com
        self.rec = rec
        self.preprocess()
        self.other_tensors()
        self.init_model()


# ===================================================================
# preprocess
# ===================================================================
    def preprocess(self):
        self.train_label = []
        self.test_label = []
        if self.args.type == "YCbCr" : input_setup = input_setup_demo
        elif self.args.type == "RGB" : input_setup = input_setup_demo
        else : input_setup = input_setup_demo

        # scale augmentation
        scale_temp = self.args.scale
        for s in [2]:
            self.args.scale = s
            _, train_label_ = input_setup(self.args, mode="train")
            self.train_label.extend(train_label_)
        self.args.scale = scale_temp

        # augmentation (rotation, miror flip)
        self.train_label = augumentation(self.train_label)

        # setup test data
        _, self.test_label = input_setup(self.args, mode="test")




    def other_tensors(self):
        with tf.variable_scope("trainer") as scope:
            # com optimizer
            vars_com = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="com")
            self.global_step_com = tf.Variable(0, trainable=False, name="global_step_com")
            self.loss_com = tf.reduce_mean(tf.abs(self.rec.pred - self.com.labels)) #L1 is betther than L2
            self.learning_rate_com = tf.maximum(tf.train.exponential_decay(self.args.base_lr_com, self.global_step_com,
                                                            len(self.train_label)//self.args.batch_size * self.args.lr_step_size_com,
                                                            self.args.lr_decay_rate_com, staircase=False),self.args.min_lr_com) #stair case showed better result
            self.train_op_com = tf.train.AdamOptimizer(self.learning_rate_com).minimize(self.loss_com,global_step=self.global_step_com, var_list=vars_com)


            # rec optimizer
            vars_rec = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="rec")
            self.global_step_rec = tf.Variable(0, trainable=False, name="global_step_rec")
            self.loss_rec = tf.reduce_mean(tf.abs(self.rec.pred_co - self.rec.labels)) #L1 is betther than L2
            self.learning_rate_rec = tf.maximum(tf.train.exponential_decay(self.args.base_lr_rec, self.global_step_rec,
                                                            len(self.train_label)//self.args.batch_size * self.args.lr_step_size_rec,
                                                            self.args.lr_decay_rate_rec, staircase=False),self.args.min_lr_rec) #stair case showed better result
            self.train_op_rec = tf.train.AdamOptimizer(self.learning_rate_rec).minimize(self.loss_rec,global_step=self.global_step_rec, var_list=vars_rec)



            # tensor board
            self.summary_writer = tf.summary.FileWriter("./board", self.sess.graph)



    def init_model(self):
        vars_trainer = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="trainer")
        self.sess.run(tf.variables_initializer(var_list=vars_trainer))

        self.com.cpkt_save(self.args.checkpoint_dir_com, self.global_step_com)
        self.rec.cpkt_save(self.args.checkpoint_dir_rec, self.global_step_rec)
        pass




# ==========================================================
# train
# ==========================================================
    def train(self):
        self.test()
        print("Training...")
        start_time = time.time()


        for ep in range(self.args.epoch):
            # =============== shuffle and prepare batch images ============================
            seed = int(time.time())
            np.random.seed(seed); np.random.shuffle(self.train_label)

            '''
            result = self.rec.pred.eval(feed_dict = {self.rec.com.images: self.train_label[0:self.args.batch_size]})[0]
            print(result)
            print(result.shape)
            '''

            #================ train rec ===================================================
            batch_idxs = len(self.train_label) // self.args.batch_size
            for idx in tqdm(range(0, batch_idxs)):
                batch_labels = self.train_label[idx * self.args.batch_size: (idx + 1) * self.args.batch_size]


                batch_labels_ds_co = []
                for b in batch_labels:
                    b = ycbcr2rgb(self.com.inference(b))
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpgqfactor]
                    result, encimg = cv2.imencode('.jpg', (b*255)[...,::-1], encode_param)
                    img_     = cv2.imdecode(encimg, 1)[...,::-1]
                    img = rgb2ycbcr((img_/255).astype(np.float32))
                    batch_labels_ds_co.append(img)


                batch_labels_ds_co = np.array(batch_labels_ds_co)
                batch_labels_ds_co = np.expand_dims(np.array(batch_labels_ds_co)[:,:,:,0],-1)
                batch_labels = np.expand_dims(np.array(batch_labels)[:,:,:,0],-1)


                feed = {self.rec.images_co: batch_labels_ds_co, self.rec.labels:batch_labels, self.rec.train_phase:1, self.com.train_phase:1}
                _, err_rec, lr_rec = self.sess.run( [self.train_op_rec, self.loss_rec, self.learning_rate_rec], feed_dict=feed)


            #================ train com ===================================================
            for idx in tqdm(range(0, batch_idxs)):
                batch_labels = self.train_label[idx * self.args.batch_size: (idx + 1) * self.args.batch_size]
                batch_labels = np.expand_dims(np.array(batch_labels)[:,:,:,0],-1)

                feed = {self.rec.com.images: batch_labels, self.com.labels:batch_labels, self.rec.train_phase:1, self.com.train_phase:1}
                _, err_com, lr_com = self.sess.run([self.train_op_com, self.loss_com, self.learning_rate_com],  feed_dict=feed)


            #=============== print log =====================================================
            if ep % 1 == 0:
                print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_rec: [%.8f], lr: [%.8f]" \
                      % ((ep + 1), self.global_step_rec.eval(), time.time() - start_time, np.mean(err_rec), lr_rec))
                print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_com: [%.8f], lr: [%.8f]" \
                      % ((ep + 1), self.global_step_com.eval(), time.time() - start_time, np.mean(err_com), lr_com))
                self.test()


            #================ save checkpoints ===============================================
            if ep % self.args.save_period_rec == 0:
                self.rec.cpkt_save(self.args.checkpoint_dir_rec, ep + 1)
            if ep % self.args.save_period_com == 0:
                self.com.cpkt_save(self.args.checkpoint_dir_com, ep + 1)







# ==========================================================
# test
# ==========================================================
    def test(self):
        print("Testing...")
        psnrs_preds = []
        ssims_preds = []

        preds = []
        labels = []
        images = []

        for idx in range(0, len(self.test_label)):
            img_ori = np.array(self.test_label[idx]) #none,none,3

            '''
            #=== bicubic ============
            cv2.imwrite(os.path.join(self.args.result_dir, str(idx) + "oribicdownTemp" + str(self.args.jpgqfactor)+ ".jpg"),
                        (imresize(img_ori,0.5) * 255)[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpgqfactor])

            btemp = cv2.imread(os.path.join(self.args.result_dir, str(idx) + "oribicdownTemp" + str(self.args.jpgqfactor)+ ".jpg"))[...,
            ::-1] / 255
            cv2.imwrite(os.path.join(self.args.result_dir, str(idx) + "oribicrecTemp" + str(self.args.jpgqfactor)+ ".jpg"),
                        (imresize(btemp,2) * 255)[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpgqfactor])
            cv2.imwrite(os.path.join(self.args.result_dir, str(idx) + "oribicrecTemp.png"),
                (imresize(btemp, 2) * 255)[..., ::-1])

            #======================================


            #=== original =====
            for f in [5,10,20,40,50,60,80,100]:
                cv2.imwrite(os.path.join(self.args.result_dir,  str(idx) + "oriTemp" +str(f) +".jpg"), (img_ori * 255)[..., ::-1],[int(cv2.IMWRITE_JPEG_QUALITY),f])
            cv2.imwrite(os.path.join(self.args.result_dir, str(idx) + "oriTemp.PNG"), (img_ori * 255)[..., ::-1])
            #==================

            #== model ==================
            result = self.com.inference(img_ori)
            cv2.imwrite(os.path.join(self.args.result_dir, str(idx) + "comTemp" +  str(self.args.jpgqfactor)+ ".bmp"), (result * 255)[..., ::-1])
            cv2.imwrite(os.path.join(self.args.result_dir,str(idx)+ "comTemp"+str(self.args.jpgqfactor)+".jpg"), (result*255)[...,::-1], [int(cv2.IMWRITE_JPEG_QUALITY),self.args.jpgqfactor])

            result = cv2.imread(os.path.join(self.args.result_dir,str(idx)+ "comTemp"+str(self.args.jpgqfactor)+".jpg"))[...,::-1]
            result = self.rec.inference(result[:,:,0:self.args.c_dim])


            cv2.imwrite(os.path.join(self.args.result_dir,str(idx)+ "recTemp"+str(self.args.jpgqfactor)+".bmp"), (result*255)[...,::-1])
            '''

            # === original =====
            for f in [5, 10, 20, 40, 50, 60, 80, 100]:
                cv2.imwrite(os.path.join(self.args.result_dir, str(idx) + "ori" + str(f) + ".jpg"),
                            (ycbcr2rgb(img_ori)*255)[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), f])
            cv2.imwrite(os.path.join(self.args.result_dir, str(idx) + "ori.PNG"), (ycbcr2rgb(img_ori)*255)[..., ::-1])
            # ==================
            result = self.com.inference(img_ori)

            cv2.imwrite(os.path.join(self.args.result_dir, str(idx) + "com" + str(self.args.jpgqfactor) + ".png"), (ycbcr2rgb(result)*255)[..., ::-1])
            cv2.imwrite(os.path.join(self.args.result_dir, str(idx) + "com" + str(self.args.jpgqfactor) + ".jpg"), (ycbcr2rgb(result)*255)[...,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpgqfactor])
            result = (cv2.imread(os.path.join(self.args.result_dir, str(idx) + "com" + str(self.args.jpgqfactor) + ".jpg"))[...,::-1]).astype(np.float32)/255



            result = self.rec.inference(rgb2ycbcr(result))
            cv2.imwrite(os.path.join(self.args.result_dir,str(idx)+ "rec"+str(self.args.jpgqfactor)+".bmp"), (ycbcr2rgb(result)*255)[...,::-1])



            preds.append(result)
            labels.append(img_ori)


        # cal PSNRs for each images upscaled from different depths
        for i in range(len(self.test_label)):
            if len(np.array(labels[i]).shape)==3 : labels[i] = np.array(labels[i])[:,:,0]
            if len(np.array(preds[i]).shape)==3 : preds[i] = np.array(preds[i])[:,:,0]
            psnrs_preds.append(psnr(labels[i], preds[i], max=1.0, scale=self.args.scale))
            ssims_preds.append(ssim(labels[i], preds[i], max=1.0, scale=self.args.scale))

        # print evalutaion results
        print("===================================================================================")
        print("PSNR: " + str(round(np.mean(np.clip(psnrs_preds, 0, 100)), 3)) + "dB")
        print("SSIM: " + str(round(np.mean(np.clip(ssims_preds, 0, 100)), 5)))
        print("===================================================================================")