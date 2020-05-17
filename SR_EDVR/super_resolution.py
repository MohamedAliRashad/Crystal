import glob

import cv2
import torch
from torch.hub import load_state_dict_from_url
from tqdm import tqdm

import SR_EDVR.EDVR_arch as EDVR_arch
import SR_EDVR.utils.data_utils as data_util
import SR_EDVR.utils.util as util
from utils import *

Weights = {
    "EDVR_REDS_SR_L": "https://drive.google.com/uc?export=download&id=1PYULZmtpsmY4Wx8M9f4owdLIwcwQFEmi",
    "EDVR_REDS_deblur_L": "https://drive.google.com/uc?export=download&id=1ZCl0aU8isEnUCsUYv9rIZZQrGo7vBFUH",
    "EDVR_REDS_deblurcomp_L": "https://drive.google.com/uc?export=download&id=1SGVehpZt4WL_X8Jh6blyqmHpc8DdImgv",
    "EDVR_REDS_SRblur_L": "https://drive.google.com/uc?export=download&id=18ev7Zx_10-C8-0tAVAe_BpYeLHpr_ChE",
    "EDVR_Vimeo90K_SR_L": "https://drive.google.com/uc?export=download&id=1I7x87ee3E1DoFVgMxX09nfIb2tdUdE3x",
    "EDVR_REDS_SR_Stage2": "https://drive.google.com/uc?export=download&id=1kfArevFT8hzbUT2QWXFmUl983LTebQGP",
    "EDVR_REDS_deblur_Stage2": "https://drive.google.com/uc?export=download&id=1Y1y6v40dL74Kgf5fxbGd0QC010LFCBYz",
    "EDVR_REDS_deblurcomp_Stage2": "https://drive.google.com/uc?export=download&id=1G466gQ1rRl8MUKSEbtaR0U5xgIWdsG66",
    "EDVR_REDS_SRblur_Stage2": "https://drive.google.com/uc?export=download&id=13c-VxMdf8h7MGX-_y4xamxo1hhOMYzsH",
}


class Super_Resolution:
	def __init__(self, data_mode, inframes, outframes, chunk_size=100, finetune=True):
		# options: vid4 | sharp_bicubic | blur_bicubic | blur | blur_comps
		self.data_mode = data_mode
		self.inframes = Path(inframes)
		self.outframes = Path(outframes)
		self.chunk_size = chunk_size
		self.fine_tune_stage2 = finetune

	def __edvrPredict(self, chunk_size, stage):
		device = torch.device("cuda")
		os.environ["CUDA_VISIBLE_DEVICES"] = "0"
		data_mode = self.data_mode
		# Vid4: SR
		# REDS4: sharp_bicubic (SR-clean), blur_bicubic (SR-blur);
		#        blur (deblur-clean), blur_comp (deblur-compression).
		self.stage = stage  # 1 or 2, use two stage strategy for REDS dataset.
		flip_test = False
		################## Model ##################
		model_name = None
		if data_mode == "Vid4":
			if stage == 1:
				model_name = "EDVR_Vimeo90K_SR_L"
			else:
				raise ValueError("Vid4 does not support stage 2.")
		elif data_mode == "sharp_bicubic":
			if stage == 1:
				model_name = "EDVR_REDS_SR_L"
			else:
				model_name = "EDVR_REDS_SR_Stage2"
		elif data_mode == "blur_bicubic":
			if stage == 1:
				model_name = "EDVR_REDS_SRblur_L"
			else:
				model_name = "EDVR_REDS_SRblur_Stage2"
		elif data_mode == "blur":
			if stage == 1:
				model_name = "EDVR_REDS_deblur_L"
			else:
				model_name = "EDVR_REDS_deblur_Stage2"
		elif data_mode == "blur_comp":
			if stage == 1:
				model_name = "EDVR_REDS_deblurcomp_L"
			else:
				model_name = "EDVR_REDS_deblurcomp_Stage2"
		else:
			raise NotImplementedError

		print("Model Used: ", model_name)

		if data_mode == "Vid4":
			N_in = 7  # use N_in images to restore one HR image
		else:
			N_in = 5

		predeblur, HR_in = False, False
		back_RBs = 40
		if data_mode == "blur_bicubic":
			predeblur = True
		if data_mode == "blur" or data_mode == "blur_comp":
			predeblur, HR_in = True, True
		if stage == 2:
			HR_in = True
			back_RBs = 20
		model = EDVR_arch.EDVR(
			128, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in
		)

		#### evaluation
		crop_border = 0
		border_frame = N_in // 2  # border frames when evaluate
		# temporal padding mode
		if data_mode in ("Vid4", "sharp_bicubic"):
			padding = "new_info"
		else:
			padding = "replicate"
		save_imgs = True

		save_folder = str(
			outframes_root.absolute()
		)  ## intermediate frames , not a video , not an end result
		util.mkdirs(save_folder)

		#### set up the models
		state_dict = load_state_dict_from_url(Weights[model_name])
		model.load_state_dict(state_dict, strict=True)
		model.eval()
		model = model.to(device)

		avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
		subfolder_name_l = []
		# remove old video_subfolder if exists
		remove_subfolders()
		subfolder_l = sorted(next(os.walk(self.inframes))[1])
		subfolder_l = [osp.join(self.inframes, f) for f in subfolder_l]
		# for each subfolder
		for subfolder in subfolder_l:  # [train_station]
			subfolder_name = osp.basename(subfolder)
			subfolder_name_l.append(subfolder_name)
			save_subfolder = osp.join(
				save_folder, subfolder_name
			)  # outframes/train_station
			img_path_l = sorted(glob(osp.join(subfolder, "*")))  ## all 1600 frames
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
				subSubFolder_l = sorted(glob(osp.join(subSubFolder, "*")))
				max_idx = len(subSubFolder_l)
				avg_psnr, avg_psnr_border, avg_psnr_center, N_border, N_center = (
					0,
					0,
					0,
					0,
					0,
				)

				# process each image
				for img_idx, img_path in tqdm(enumerate(subSubFolder_l)):
					img_name = osp.splitext(osp.basename(img_path))[0]
					select_idx = data_util.index_generation(
						img_idx, max_idx, N_in, padding=padding
					)
					imgs_in = (
						imgs_LQ.index_select(0, torch.LongTensor(select_idx))
						.unsqueeze(0)
						.to(device)
					)

					if flip_test:
						output = util.flipx4_forward(model, imgs_in)
					else:
						output = util.single_forward(model, imgs_in)
					output = util.tensor2img(output.squeeze(0))

					if save_imgs:
						cv2.imwrite(
							osp.join(save_subfolder, "{}.jpg".format(img_name)), output
						)

	def edvr_video(self):

		# process frames: stage 1
		self.__edvrPredict(self.chunk_size, 1)
		# fine-tune stage 2

		if self.fine_tune_stage2 and self.data_mode != "Vid4":
			# move the stage 1 processed frames over
			moveProcessedFrames()
			# process again
			self.__edvrPredict(self.chunk_size, 2)


if __name__ == "__main__":
	pass