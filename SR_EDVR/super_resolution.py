import glob

import cv2
import torch
from torch.hub import load_state_dict_from_url
from tqdm import tqdm
import os.path as osp
from pathlib import Path

from .EDVR_arch import EDVR
from .utils.data_utils import index_generation, read_img_seq
from .utils.util import flipx4_forward, mkdirs, single_forward, tensor2img, preProcess

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


def SuperResolution(inframes, outframes, stage, data_mode, use_gpu=True):
    """
	Perform a Super Resolution step on a frames folder

	Args
	----
		inframes(str, Path): folder with the frames to enhance
		outframes(str, Path): the output directory
		stage(int): determine the stage used (1 or 2)
		data_mode(str): the process wanted
			Vid4: SR
			REDS4: sharp_bicubic (SR-clean), blur_bicubic (SR-blur);
					blur (deblur-clean), blur_comp (deblur-compression).
	"""

    flip_test = False
    inframes = Path(inframes)
    outframes = str(outframes)
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

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

    # Initialize the model
    model = EDVR(128, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in)

    #### evaluation
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    if data_mode in ("Vid4", "sharp_bicubic"):
        padding = "new_info"
    else:
        padding = "replicate"
    save_imgs = True

    #### set up the models
    state_dict = load_state_dict_from_url(Weights[model_name], model_dir=model_name)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)

    img_path_l = sorted(inframes.glob("*"))

    # preprocess images (needed for blurred models)
    if predeblur:
        preProcess(img_path_l, 16)
    else:
        preProcess(img_path_l, 4)

    imgs_LQ = read_img_seq(inframes)
    max_idx = len(img_path_l)

    # process each image
    for img_idx, img_path in enumerate(tqdm(img_path_l)):
        img_name = osp.splitext(osp.basename(img_path))[0]
        select_idx = index_generation(img_idx, max_idx, N_in, padding=padding)
        imgs_in = (
            imgs_LQ.index_select(0, torch.LongTensor(select_idx))
            .unsqueeze(0)
            .to(device)
        )

        if flip_test:
            output = flipx4_forward(model, imgs_in)
        else:
            output = single_forward(model, imgs_in)
        output = tensor2img(output.squeeze(0))

        if save_imgs:
            cv2.imwrite(osp.join(outframes, "{}.jpg".format(img_name)), output)


if __name__ == "__main__":
    pass
