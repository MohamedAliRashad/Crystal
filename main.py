import argparse
#from utils import video2frames, frames2video, load_DAIN, infer_DAIN, multiply_nameby2
from utils import *
import os
import torch
from pathlib import Path
from shutil import rmtree
from SR_EDVR.super_resolution import Super_Resolution


def main(video_path):

    TEMP_FOLDER = Path(osp.join(inframes_root, Path(video_path).stem))
    if TEMP_FOLDER.exists():
        rmtree(TEMP_FOLDER)

    torch.cuda.empty_cache()
    model = load_DAIN()
    video_path = Path(video_path)

    TEMP_FOLDER.mkdir(parents=True, exist_ok=True)
    meta_data = video2frames(video_path, TEMP_FOLDER)
    multiply_nameby2(TEMP_FOLDER)

    meta_data = infer_DAIN(model, meta_data, TEMP_FOLDER)
    sr = Super_Resolution('sharp_bicubic')
    sr.edvr_video(meta_data)
    frames2video(result_folder, outframes_root , meta_data)

    rmtree(TEMP_FOLDER)

    return meta_data["name"] + ".mp4"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_file")
    args = parser.parse_args()

    main(args.video_file)
