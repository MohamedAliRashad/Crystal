import glob
import cv2
import torch
#import numpy as np
from tqdm import tqdm
import utils.util as util
import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch
from utils.video_utils import *



class EDVR():

	def __init__(self, data_mode, video_path, save_path,chunk_size=100,finetune=True):
		self.pretrained_models = Path('../experiments/pretrained_models')
		self.data_mode = data_mode # options: vid4 | sharp_bicubic | blur_bicubic | blur | blur_comp
		#self.fine_tune_stage2 = True
		self.video_path = video_path
		self.save_path = save_path
		self.chunk_size = chunk_size
		self.fine_tune_stage2 = finetune



	def __edvrPredict(self, chunk_size,stage):
	  device = torch.device('cuda')
	  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	  data_mode = self.data_mode  
	  # Vid4: SR
	  # REDS4: sharp_bicubic (SR-clean), blur_bicubic (SR-blur);
	  #        blur (deblur-clean), blur_comp (deblur-compression).
	  self.stage = stage  # 1 or 2, use two stage strategy for REDS dataset.
	  flip_test = False
	  ############################################################################
	  #### model
	  if data_mode == 'Vid4':
	      if stage == 1:
	          model_path = self.pretrained_models / 'EDVR_Vimeo90K_SR_L.pth'
	      else:
	          raise ValueError('Vid4 does not support stage 2.')
	  elif data_mode == 'sharp_bicubic':
	      if stage == 1:
	          model_path = self.pretrained_models / 'EDVR_REDS_SR_L.pth'
	      else:
	          model_path = self.pretrained_models / 'EDVR_REDS_SR_Stage2.pth'
	  elif data_mode == 'blur_bicubic':
	      if stage == 1:
	          model_path = self.pretrained_models / 'EDVR_REDS_SRblur_L.pth'
	      else:
	          model_path = self.pretrained_models / 'EDVR_REDS_SRblur_Stage2.pth'
	  elif data_mode == 'blur':
	      if stage == 1:
	          model_path = self.pretrained_models / 'EDVR_REDS_deblur_L.pth'
	      else:
	          model_path = self.pretrained_models / 'EDVR_REDS_deblur_Stage2.pth'
	  elif data_mode == 'blur_comp':
	      if stage == 1:
	          model_path = self.pretrained_models / 'EDVR_REDS_deblurcomp_L.pth'
	      else:
	          model_path = self.pretrained_models / 'EDVR_REDS_deblurcomp_Stage2.pth'     
	  else:
	      raise NotImplementedError
	  print('Model Used: ', model_path)
	  
	  if data_mode == 'Vid4':
	      N_in = 7  # use N_in images to restore one HR image
	  else:
	      N_in = 5

	  predeblur, HR_in = False, False
	  back_RBs = 40
	  if data_mode == 'blur_bicubic':
	      predeblur = True
	  if data_mode == 'blur' or data_mode == 'blur_comp':
	      predeblur, HR_in = True, True
	  if stage == 2:
	      HR_in = True
	      back_RBs = 20
	  model = EDVR_arch.EDVR(128, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)

	  #### dataset
	  test_dataset_folder = inframes_root

	  #### evaluation
	  crop_border = 0
	  border_frame = N_in // 2  # border frames when evaluate
	  # temporal padding mode
	  if data_mode in ('Vid4','sharp_bicubic'):
	      padding = 'new_info'
	  else:
	      padding = 'replicate'
	  save_imgs = True

	  save_folder = str(outframes_root.absolute())
	  util.mkdirs(save_folder)

	  #### set up the models
	  model.load_state_dict(torch.load(model_path), strict=True)
	  model.eval()
	  model = model.to(device)

	  avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
	  subfolder_name_l = []
	  # remove old video_subfolder if exists
	  remove_subfolders()
	  subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))

	  # for each subfolder
	  for subfolder in subfolder_l:
	      subfolder_name = osp.basename(subfolder)
	      subfolder_name_l.append(subfolder_name)
	      save_subfolder = osp.join(save_folder, subfolder_name)

	      img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
	      if save_imgs:
	          util.mkdirs(save_subfolder)
	          purge_images(save_subfolder)

	      # preprocess images (needed for blurred models)
	      if predeblur:
	        preProcess(img_path_l, 16)
	      else:
	        preProcess(img_path_l, 4)
	      # make even more subfolders
	      subFolderList = make_subfolders(img_path_l, chunk_size)

	      #### read LQ and GT images in chunks of 1000
	      for subSubFolder in subFolderList:
	        clean_mem()
	        imgs_LQ = data_util.read_img_seq(subSubFolder)
	        subSubFolder_l = sorted(glob.glob(osp.join(subSubFolder, '*')))
	        max_idx = len(subSubFolder_l)
	        avg_psnr, avg_psnr_border, avg_psnr_center, N_border, N_center = 0, 0, 0, 0, 0

	        # process each image
	        for img_idx, img_path in tqdm(enumerate(subSubFolder_l)):
	            img_name = osp.splitext(osp.basename(img_path))[0]
	            select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
	            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

	            if flip_test:
	                output = util.flipx4_forward(model, imgs_in)
	            else:
	                output = util.single_forward(model, imgs_in)
	            output = util.tensor2img(output.squeeze(0))

	            if save_imgs:
	                cv2.imwrite(osp.join(save_subfolder, '{}.jpg'.format(img_name)), output)
	                # print('Saved Image:', str(osp.join(save_subfolder, '{}.jpg'.format(img_name))))




	def edvr_video(self):

#		extract_raw_frames(self.video_path)		## for if inframes doesn't yet exist

		# process frames: stage 1 
		self.__edvrPredict(self.chunk_size,1)
		# fine-tune stage 2
		if self.fine_tune_stage2:
		# move the stage 1 processed frames over
			moveProcessedFrames()
		# process again
			self.__edvrPredict(self.chunk_size, 2)

		# build back video
		build_video(self.video_path, self.save_path)




if __name__ == '__main__':
	enhancer = Super_Resolution('blur',Path('/content/video.mp4'),Path('/content/EDVR/codes/video/'))
	enhancer.edvr_video()
	pass
