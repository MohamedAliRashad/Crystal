import argparse
from utils import video2frames, build_video, load_DAIN, infer_DAIN, purge_images, clean_mem
import os
from pathlib import Path
from shutil import rmtree
from SR_EDVR.super_resolution import SuperResolution

TEMP_FOLDER1 = Path('./tmp1/')
TEMP_FOLDER2 = Path('./tmp2/')

def main(video_path, output_path):

    # Prepare the environment for the process
    clean_mem()
    video_path = Path(video_path)
    output_path = Path(output_path)

    if TEMP_FOLDER1.exists():
        rmtree(TEMP_FOLDER1)

    if TEMP_FOLDER2.exists():
        rmtree(TEMP_FOLDER2)

    TEMP_FOLDER1.mkdir(parents=True, exist_ok=True)
    TEMP_FOLDER2.mkdir(parents=True, exist_ok=True)
    
    # Initiliaze the Models
    model = load_DAIN()

    # Extract frames from video
    meta_data = video2frames(video_path, TEMP_FOLDER1)
    
    # Step1: Double the frame rate
    meta_data = infer_DAIN(model, meta_data, TEMP_FOLDER1, TEMP_FOLDER2)
    purge_images(TEMP_FOLDER1)

    # Step2: Run Stage 1 of EDVR
    SuperResolution(TEMP_FOLDER2, TEMP_FOLDER1, 1, "sharp_bicubic")
    purge_images(TEMP_FOLDER2)

    # Step3: Run Stage 2 for refinement
    SuperResolution(TEMP_FOLDER1, TEMP_FOLDER2, 2, "sharp_bicubic")

    # Recreate the video
    build_video(TEMP_FOLDER2, output_path, meta_data)

    # Clean Everything
    clean_mem()
    rmtree(TEMP_FOLDER1)
    rmtree(TEMP_FOLDER2)

    return meta_data["name"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_file")
    parser.add_argument("--output", "-o", default="./downloads/")
    args = parser.parse_args()

    main(args.video_file, args.output)
