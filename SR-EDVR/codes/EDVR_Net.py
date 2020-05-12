from utils.video_utils import *

import torch
import utils.util as util
import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch
import gc




class EDVR_net():

    def __init__(self,data_mode,finetune_stage2,chunk_size):

        self.device = torch.device('cuda')
        self.stage2_model_path = None
        self.data_mode = data_mode
        if data_mode == 'Vid4':
            self.model_path = pretrained_models / 'EDVR_Vimeo90K_SR_L.pth'
        elif data_mode == 'sharp_bicubic':
            self.model_path = pretrained_models / 'EDVR_REDS_SR_L.pth'
            if finetune_stage2:
                self.stage2_model_path = pretrained_models / 'EDVR_REDS_SR_Stage2.pth'
        elif data_mode == 'blur_bicubic':
            self.model_path = pretrained_models / 'EDVR_REDS_SRblur_L.pth'
            if finetune_stage2:
                self.stage2_model_path = pretrained_models / 'EDVR_REDS_SRblur_Stage2.pth'
        elif data_mode == 'blur':
            self.model_path = pretrained_models / 'EDVR_REDS_deblur_L.pth'
            if finetune_stage2:
                self.stage2_model_path = pretrained_models / 'EDVR_REDS_deblur_Stage2.pth'
        elif data_mode == 'blur_comp':
            self.model_path = pretrained_models / 'EDVR_REDS_deblurcomp_L.pth'
            if finetune_stage2:
                self.stage2_model_path = pretrained_models / 'EDVR_REDS_deblurcomp_Stage2.pth'     
        else:
            raise NotImplementedError
        print('Model Used: ', self.model_path)

        if data_mode == 'Vid4':
            self.N_in = 7  # use self.N_in images to restore one HR image
        else:
            self.N_in = 5

        self.predeblur, self.HR_in = False, False
        self.back_RBs = 40
        if data_mode == 'blur_bicubic':
            self.predeblur = True
        if data_mode == 'blur' or data_mode == 'blur_comp':
            self.predeblur, self.HR_in = True, True



        self.test_dataset_folder = osp.join(parent_dir,'video/inframes')

        #### evaluation
        self.crop_border = 0
        self.border_frame = self.N_in // 2  # border frames when evaluate
        # temporal padding mode
        if data_mode in ('Vid4','sharp_bicubic'):
            self.padding = 'new_info'
        else:
            self.padding = 'replicate'
        save_imgs = True

        self.save_folder = osp.join(parent_dir,'video/outframes')
        util.mkdirs(self.save_folder)

        
    def load_model(self,stage=1):
        if stage == 1 :
            self.model = EDVR_arch.EDVR(128, self.N_in, 8, 5, self.back_RBs, predeblur=self.predeblur, HR_in=HR_in)
            self.model.load_state_dict(torch.load(self.model_path), strict=True)
        else:
            HR_in = True
            back_RBs = 20
            self.model = EDVR_arch.EDVR(128, self.N_in, 8, 5, back_RBs, predeblur=self.predeblur, HR_in=HR_in)
            self.model.load_state_dict(torch.load(self.stage2_model_path), strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)



    def model_feed(self):
        pass
