import gc
import os
import os.path as osp
import re
import warnings
from pathlib import Path

import cv2
import ffmpeg
import imageio
import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")


def clean_mem():
    torch.cuda.empty_cache()
    gc.collect()


def video2frames(video_path, output_dir):

    reader = imageio.get_reader(video_path)
    meta_data = reader.get_meta_data()
    fps = meta_data["fps"]
    size = meta_data["size"]

    extract_raw_frames(video_path, output_dir)
    name = Path(video_path).name

    return {"fps": fps, "name": name, "size": size, "video_path": video_path}


def load_DAIN():

    # Let the magic happen
    from DAIN.DAIN import DAIN

    module = DAIN()

    # load the weights online
    from torch.hub import load_state_dict_from_url

    state_dict = load_state_dict_from_url(
        "http://vllab1.ucmerced.edu/~wenbobao/DAIN/best.pth"
    )
    module.load_state_dict(state_dict)

    return module


def infer_DAIN(model, meta_data, inframes, outframes, use_gpu=True, scale_precent=100):
    """
    Interpolate frames using DAIN

    Args
    ---
        model(nn.Module): DAIN Module for processing
        meta_data(dict): info about the video like (size and fps)
        inframes(str, Path): input frames directory 
        outframes(str, Path): output frames directory 
        use_gpu(bool): choosing between GPU and CPU
        scale(int): how much to reduce the original size of the frames (it's reversed at the end)

    """
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
    model.to(device)

    inframes = Path(inframes)
    outframes = str(outframes)

    frames = sorted(inframes.glob("*"))

    width = int(meta_data["size"][0] * scale_precent / 100)
    height = int(meta_data["size"][1] * scale_precent / 100)
    dim = (width, height)

    model.eval()

    j = 0
    for i in tqdm(range(len(frames) - 1)):

        image1 = cv2.resize(
            imageio.imread(frames[i]), dim, interpolation=cv2.INTER_AREA
        )
        image2 = cv2.resize(
            imageio.imread(frames[i + 1]), dim, interpolation=cv2.INTER_AREA
        )

        X0 = torch.from_numpy(np.transpose(image1, (2, 0, 1)).astype("float32") / 255.0)
        X1 = torch.from_numpy(np.transpose(image2, (2, 0, 1)).astype("float32") / 255.0)
        y_ = torch.FloatTensor()

        intWidth = X0.size(2)
        intHeight = X0.size(1)
        channel = X0.size(0)

        if intWidth != ((intWidth >> 7) << 7):
            intWidth_pad = ((intWidth >> 7) + 1) << 7  # more than necessary
            intPaddingLeft = int((intWidth_pad - intWidth) / 2)
            intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
        else:
            intWidth_pad = intWidth
            intPaddingLeft = 32
            intPaddingRight = 32

        if intHeight != ((intHeight >> 7) << 7):
            intHeight_pad = ((intHeight >> 7) + 1) << 7  # more than necessary
            intPaddingTop = int((intHeight_pad - intHeight) / 2)
            intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
        else:
            intHeight_pad = intHeight
            intPaddingTop = 32
            intPaddingBottom = 32

        pader = torch.nn.ReplicationPad2d(
            [intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom]
        )

        torch.set_grad_enabled(False)
        X0 = torch.unsqueeze(X0, 0)
        X1 = torch.unsqueeze(X1, 0)

        X0 = pader(X0).to(device)
        X1 = pader(X1).to(device)

        y_s, offset, filter = model(torch.stack((X0, X1), dim=0))

        y_ = y_s[1]

        X0 = X0.data.cpu().numpy()
        y_ = y_.data.cpu().numpy()
        offset = [offset_i.data.cpu().numpy() for offset_i in offset]
        filter = (
            [filter_i.data.cpu().numpy() for filter_i in filter]
            if filter[0] is not None
            else None
        )
        X1 = X1.data.cpu().numpy()

        X0 = np.transpose(
            255.0
            * X0.clip(0, 1.0)[
                0,
                :,
                intPaddingTop : intPaddingTop + intHeight,
                intPaddingLeft : intPaddingLeft + intWidth,
            ],
            (1, 2, 0),
        )
        y_ = np.transpose(
            255.0
            * y_.clip(0, 1.0)[
                0,
                :,
                intPaddingTop : intPaddingTop + intHeight,
                intPaddingLeft : intPaddingLeft + intWidth,
            ],
            (1, 2, 0),
        )
        offset = [
            np.transpose(
                offset_i[
                    0,
                    :,
                    intPaddingTop : intPaddingTop + intHeight,
                    intPaddingLeft : intPaddingLeft + intWidth,
                ],
                (1, 2, 0),
            )
            for offset_i in offset
        ]
        filter = (
            [
                np.transpose(
                    filter_i[
                        0,
                        :,
                        intPaddingTop : intPaddingTop + intHeight,
                        intPaddingLeft : intPaddingLeft + intWidth,
                    ],
                    (1, 2, 0),
                )
                for filter_i in filter
            ]
            if filter is not None
            else None
        )
        X1 = np.transpose(
            255.0
            * X1.clip(0, 1.0)[
                0,
                :,
                intPaddingTop : intPaddingTop + intHeight,
                intPaddingLeft : intPaddingLeft + intWidth,
            ],
            (1, 2, 0),
        )

        imageio.imsave(
            os.path.join(outframes, str(j).zfill(6) + ".jpg"),
            cv2.resize(image1, meta_data["size"], interpolation=cv2.INTER_AREA),
        )
        imageio.imsave(
            os.path.join(outframes, str(j + 1).zfill(6) + ".jpg"),
            cv2.resize(
                np.round(y_).astype(np.uint8),
                meta_data["size"],
                interpolation=cv2.INTER_AREA,
            ),
        )

        j = j + 2

    imageio.imsave(
        os.path.join(outframes, str(j).zfill(6) + ".jpg"),
        cv2.resize(image2, meta_data["size"], interpolation=cv2.INTER_AREA),
    )
    meta_data["fps"] = meta_data["fps"] * 2

    return meta_data


def download_video_from_url(source_url, source_path, quality):
    import youtube_dl

    source_path = Path(source_path)
    if source_path.exists():
        source_path.unlink()

    ydl_opts = {
        "format": "bestvideo[height<={}][ext=mp4]+bestaudio[ext=m4a]/mp4".format(
            quality
        ),
        "outtmpl": str(source_path),
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([source_url])


def purge_images(dir):
    for f in os.listdir(dir):
        if re.search(".*?\.jpg", f):
            os.remove(os.path.join(dir, f))


def extract_raw_frames(video_path, save_path):
    """
    Extracts frames from a video to a specified path

    Args
    ---
        video_path(str, Path): absolute or a relative path for the video 
        save_path(str, Path): directory to extract frames in

    """

    inframes_root = Path(save_path)
    inframe_path_template = str(inframes_root / "%6d.jpg")
    ffmpeg.input(str(video_path)).output(
        str(inframe_path_template),
        format="image2",
        vcodec="mjpeg",
        qscale=0,
        start_number=0,
    ).run(capture_stdout=True)


def build_video(frames_dir, save_dir, meta_data, audio_path=None):
    """
    Construct video from frames
    
    Args
    ----
        frames_dir(str, Path): folder path to frames
        save_dir(str, Path):

    """
    save_dir = Path(save_dir)
    out_path = str(save_dir / meta_data["name"])

    # Use ffmpeg to reconstruct the video
    ffmpeg.input(
        str(frames_dir), format="image2", vcodec="mjpeg", framerate=meta_data["fps"]
    ).output(out_path, crf=17, vcodec="libx264").run(capture_stdout=True)


def get_thumbnail(video_path):
    cwd = os.getcwd()
    thumb_path = osp.join(cwd, "thumb.jpg")

    if osp.exists(thumb_path):
        os.remove(thumb_path)

    command = (
        'ffmpeg -i "'
        + video_path
        + '" -ss 3 -vf "select=gt(scene\\,0.4)" -frames:v 5 -vsync vfr -vf fps=1/6 thumb.jpg'
    )
    os.system(command)


def multiply_nameby2(frames_path):
    frames = sorted(frames_path.glob("*.jpg"))
    for frame in frames:
        x = str(int(frame.stem) * 2).zfill(5) + ".jpg"
        frame.rename(frames_path / x)
