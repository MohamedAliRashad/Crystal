import argparse
from utils import video2frames, frames2video, load_DAIN, infer_DAIN
import os
import torch
from pathlib import Path
from shutil import rmtree

TEMP_FOLDER = Path("./tmp")

def main(video_path):

    torch.cuda.empty_cache()
    model = load_DAIN()
    video_path = Path(video_path)

    TEMP_FOLDER.mkdir(parents=True, exist_ok=True)
    meta_data = video2frames(video_path, TEMP_FOLDER)

    print("Running DAIN ....")
    meta_data = infer_DAIN(model, meta_data, TEMP_FOLDER)

    frames2video(Path("./downloads/"), TEMP_FOLDER, meta_data)

    rmtree(TEMP_FOLDER)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_file")
    args = parser.parse_args()

    if TEMP_FOLDER.exists():
        rmtree(TEMP_FOLDER)

    main(args.video_file)
