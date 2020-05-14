import imageio
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import re
import os
import cv2
import numpy as np
import torch
from glob import glob

import os.path as osp
import ffmpeg
import shutil
from pathlib import Path
import youtube_dl
from PIL import Image

workfolder = Path('./video')
source_folder = workfolder / "source"
inframes_root = workfolder / "inframes"
audio_root = workfolder / "audio"
outframes_root = workfolder / "outframes"
result_folder = workfolder / "result"
source_img_path = inframes_root / "video_subfolders"


def video2frames(video_path, output_dir):

    reader = imageio.get_reader(video_path)
    meta_data = reader.get_meta_data()
    fps = meta_data['fps']
    size = meta_data['size']
    n_frames = meta_data['nframes']

    extract_raw_frames(video_path, output_dir)
    vid_path = Path(video_path)
    name = vid_path.stem
   
    return {"fps": fps, "name": name, "size": size, "video_path": video_path}

def frames2video(save_path, frames_path, meta_data): 
    # video_path = meta_data["video_path"]
    # return build_video(video_path, frames_path, save_path, meta_data) # return the path of the saved video
    os.system("ffmpeg -framerate {} -i {}.jpg {}.mp4".format(meta_data["fps"], frames_path / "%05d" ,str(save_path / meta_data["name"])))

def load_DAIN():
    # Let the magic happen
    from DAIN.DAIN import DAIN
    module = DAIN()
    
    # load the weights online
    from torch.hub import load_state_dict_from_url
    state_dict = load_state_dict_from_url("http://vllab1.ucmerced.edu/~wenbobao/DAIN/best.pth")
    module.load_state_dict(state_dict)

    return module

def infer_DAIN(model, meta_data, frames_folder):

    model.cuda()

    # frames = sorted(glob(os.path.join(frames_folder, "*.jpg")))
    frames = sorted(frames_folder.glob("*.jpg"))

    scale_precent = 100
    width = int(meta_data["size"][0] * scale_precent / 100)
    height = int(meta_data["size"][1] * scale_precent / 100)
    dim = (width, height)
    model.eval()

    # j = 0
    for i in tqdm(range(len(frames) - 1)):

        # image1 = cv2.resize(imageio.imread(frames[i]), dim, interpolation=cv2.INTER_AREA)
        # image2 = cv2.resize(imageio.imread(frames[i + 1]), dim, interpolation=cv2.INTER_AREA)
        
        image1 = imageio.imread(frames[i])
        image2 = imageio.imread(frames[i + 1])
        
        X0 = torch.from_numpy(np.transpose(image1, (2, 0, 1)).astype("float32") / 255.0).type(torch.cuda.FloatTensor)
        X1 = torch.from_numpy(np.transpose(image2, (2, 0, 1)).astype("float32") / 255.0).type(torch.cuda.FloatTensor)
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

        X0 = pader(X0).cuda()
        X1 = pader(X1).cuda()

        y_s, offset, filter = model(torch.stack((X0, X1), dim=0))

        y_ = y_s[1]

        X0 = X0.data.cpu().numpy()
        y_ = y_.data.cpu().numpy()
        offset = [offset_i.data.cpu().numpy() for offset_i in offset]
        filter = [filter_i.data.cpu().numpy() for filter_i in filter] if filter[0] is not None else None
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

        # imageio.imsave(os.path.join(frames_folder, str(j).zfill(6) + ".jpg"), cv2.resize(image1, meta_data["size"], interpolation=cv2.INTER_AREA))
        # imageio.imsave(os.path.join(frames_folder, str(j+1).zfill(6) + ".jpg"), cv2.resize(np.round(y_).astype(np.uint8), meta_data["size"], interpolation=cv2.INTER_AREA))
        imageio.imsave(os.path.join(frames_folder, str(2*i+1).zfill(5) + ".jpg"), np.round(y_).astype(np.uint8))
        # j = j + 2
        
    # imageio.imsave(os.path.join(frames_folder, str(j).zfill(6) + ".jpg"), cv2.resize(image2, meta_data["size"], interpolation=cv2.INTER_AREA))
    meta_data["fps"] = meta_data["fps"]*2

    return meta_data


def get_fps(source_path):
    probe = ffmpeg.probe(source_path)
    stream_data = next(
        (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
        None,
    )
    return stream_data['avg_frame_rate']

def download_video_from_url(source_url, source_path, quality):
    source_path = Path(source_path)
    if source_path.exists():
        source_path.unlink()

    ydl_opts = {
        'format': 'bestvideo[height<={}][ext=mp4]+bestaudio[ext=m4a]/mp4'.format(quality),
        'outtmpl': str(source_path),
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([source_url])


def preProcess(imag_path_l, multiple):
  '''Need to resize images for blurred model (needs to be multiples of 16)'''
  for img_path in imag_path_l:
    im = Image.open(img_path)
    h, w = im.size
    # resize so they are multiples of 4 or 16 (for blurred)
    h = h - h % multiple
    w = w - w % multiple
    im = im.resize((h,w))
    im.save(img_path)

def purge_images(dir):
  for f in os.listdir(dir):
    if re.search('.*?\.jpg', f):
      os.remove(os.path.join(dir, f))


def extract_raw_frames(source_path, save_path):

    inframes_root = Path(save_path)
    # inframes_folder = inframes_root / (source_path.stem)
    inframe_path_template = str(inframes_root / '%6d.jpg')
    # inframes_folder.mkdir(parents=True, exist_ok=True)
    # purge_images(inframes_folder)
    ffmpeg.input(str(source_path)).output(
        str(inframe_path_template), format='image2', vcodec='mjpeg', qscale=0, start_number=0
    ).run(capture_stdout=True)



def make_subfolders(img_path_l, chunk_size): # frames must be in subfolders for EDVR model
  i = 0
  subFolderList = []
  source_img_path.mkdir(parents=True, exist_ok=True)
  for img in img_path_l:
    if i % chunk_size == 0:
      img_path = source_img_path / str(i)
      img_path.mkdir(parents=True, exist_ok=True)
      subFolderList.append(str(img_path))
    i+=1
    img_name = osp.basename(img)
    img_path_name = img_path / img_name
    shutil.copyfile(img, img_path_name)

  return subFolderList



def remove_subfolders():
  shutil.rmtree(source_img_path, ignore_errors=True, onerror=None)


def moveProcessedFrames():
  shutil.rmtree(inframes_root)
  os.rename(outframes_root, inframes_root)




def build_video(source_path, frames_dir, save_path, meta_data):
        out_path = save_path / (
            source_path.name.replace('.mp4', '_no_audio.mp4')
        )
        outframes_root = frames_dir
        # outframes_folder = outframes_root / (source_path.stem)
        outframes_path_template = str(outframes_root / '%5d.jpg')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # if out_path.exists():
        #     out_path.unlink()
        fps = meta_data["fps"]
        print('Original FPS is: ', fps)

        ffmpeg.input(
            str(outframes_path_template),
            format='image2',
            vcodec='mjpeg',
            framerate=fps,
        ).output(str(out_path), crf=17, vcodec='libx264').run(capture_stdout=True)

        result_path = save_path 
        if result_path.exists():
            result_path.unlink()

        # making copy of non-audio version in case adding back audio doesn't apply or fails.
        shutil.copyfile(str(out_path), str(result_path))

        # adding back sound here
        audio_file = Path(str(source_path).replace('.mp4', '.aac'))
        if audio_file.exists():
            audio_file.unlink()

        os.system(
            'ffmpeg -y -i "'
            + str(source_path)
            + '" -vn -acodec copy "'
            + str(audio_file)
            + '"'
        )

        if audio_file.exists:
            os.system(
                'ffmpeg -y -i "'
                + str(out_path)
                + '" -i "'
                + str(audio_file)
                + '" -shortest -c:v copy -c:a aac -b:a 256k "'
                + str(result_path)
                + '"'
            )
        return result_path

def get_thumbnail(video_path):
	cwd = os.getcwd()
	thumb_path = osp.join(cwd, "thumb.jpg")

	if osp.exists(thumb_path):
		os.remove(thumb_path)

	command = 'ffmpeg -i "' + video_path + '" -ss 3 -vf "select=gt(scene\\,0.4)" -frames:v 5 -vsync vfr -vf fps=1/6 thumb.jpg'
	os.system(command)
    
    # if osp.exists(thumb_path):
    #     return thumb_path

    # return None

def multiply_nameby2(frames_path):
    frames = sorted(frames_path.glob("*.jpg"))
    for frame in frames:
        x = str(int(frame.stem)*2).zfill(5)+".jpg"
        frame.rename(frames_path / x)


if __name__ == "__main__":
    # print(get_thumbnail('./uploads/Testing_Deepfaking_First_Order_Motion_Model_for_Image_Animation-9RdQfzM0FR4.mkv'))
    # video2frames("./uploads/Train.mp4", "./tmp/")
    multiply_nameby2(Path("./tmp"))