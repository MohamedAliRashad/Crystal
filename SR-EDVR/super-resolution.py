import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch

import utils.util as util
import utils.data_util as data_util
import archs.EDVR_arch as EDVR_arch




class EDVR_Model():

    def __init__(self, model_path, frames_dir, save_dir):



        self.device = torch.device('cuda')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.data_mode = 'Vid4'  # Vid4 | sharp_bicubic | blur_bicubic | blur | blur_comp
        self.imgs = [] # imgs list
       
        self.stage = 1  # 1 or 2, use two stage strategy for REDS dataset.
        self.flip_test = False

        self.model_path = model_path

        if self.data_mode == 'Vid4':
            self.N_in = 7  # use N_in images to restore one HR image
        else:
            self.N_in = 5


        predeblur, HR_in = False, False

        back_RBs = 40 # or 20 

        self.model = EDVR_arch.EDVR(128, self.N_in, 8, 5, back_RBs, predeblur= predeblur, HR_in= HR_in)

        self.frames_dir = frames_dir

        ######################

        # temporal padding mode
        if self.data_mode == 'Vid4' or self.data_mode == 'sharp_bicubic':
            self.padding = 'new_info'
        else:
            self.padding = 'replicate'

        self.save_dir = save_dir


        self.load_model()






    def load_model(self):
        #### set up the models
        self.model.load_state_dict(torch.load(self.model_path), strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)




    def read_imgs(self, path):

        imgsnames_l = os.listdir(path)
        imgs_l = [os.path.join(path, im) for im in imgsnames_l]


        for im in imgs_l:
            img = cv2.imread(im, cv2.IMREAD_UNCHANGED)
            img = img.astype(np.float32) / 255.
            self.imgs.append(img)




    def process_imgs(self, imgs_l):
        imgs = np.stack(imgs_l, axis=0)
        imgs = imgs[:, :, :, [2, 1, 0]]
        imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
        print("dims are", imgs.shape)
        return imgs




    def convert_frames(self):


        crop_border = 0
        border_frame = self.N_in // 2  # border frames when evaluate
        #img_list = os.listdir(self.frames_dir)
        self.read_imgs(self.frames_dir)
        imgs_LQ = self.process_imgs(self.imgs)
        imgsnames_l = os.listdir(self.frames_dir)

        imgs_l = [os.path.join(self.frames_dir, im) for im in imgsnames_l]

        # process each image
        for img_idx, d in enumerate(imgs_l):


            img_name = osp.splitext(osp.basename(imgs_l[img_idx]))[0]
            max_idx = len(imgs_l)
            print("max id", max_idx)
            print("data", img_idx)
            select_idx = data_util.index_generation(img_idx, max_idx - 1, self.N_in, padding=self.padding)
            select_idx = 0
            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(self.device)

            if self.flip_test:
                output = util.flipx4_forward(self.model, imgs_in)
            else:
                output = util.single_forward(self.model, imgs_in)
            
            output = util.tensor2img(output.squeeze(0))

            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)

            cv2.imwrite(osp.join(self.save_dir, '{}.png'.format(img_name)), output)




if __name__ == '__main__':

    weights_path = './SR-model.pth'
    frames_dir = './framesdir'
    save_dir = './savedir'

    model = EDVR_Model(model_path = weights_path, frames_dir = frames_dir, save_dir = save_dir)

    model.convert_frames()



