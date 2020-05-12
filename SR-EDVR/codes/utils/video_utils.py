import os
import os.path as osp
import ffmpeg
import gc
import shutil
from pathlib import Path
from PIL import Image
import youtube_dl
import re



# Paths


workfolder = Path('./video').absolute()
source_folder = workfolder / "source"
inframes_root = workfolder / "inframes"
audio_root = workfolder / "audio"
outframes_root = workfolder / "outframes"
result_folder = workfolder / "result"
source_img_path = inframes_root / "video_subfolders"



def clean_mem():
    # torch.cuda.empty_cache()
    gc.collect()

def get_fps(source_path: Path) -> str:
    probe = ffmpeg.probe(str(source_path))
    stream_data = next(
        (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
        None,
    )
    return stream_data['avg_frame_rate']

def download_video_from_url(source_url, source_path: Path, quality: str):
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

def extract_raw_frames(source_path: Path):
    inframes_folder = inframes_root / (source_path.stem)
    inframe_path_template = str(inframes_folder / '%5d.jpg')
    inframes_folder.mkdir(parents=True, exist_ok=True)
    purge_images(inframes_folder)
    ffmpeg.input(str(source_path)).output(
        str(inframe_path_template), format='image2', vcodec='mjpeg', qscale=0
    ).run(capture_stdout=True)

def make_subfolders(img_path_l, chunk_size):
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
  shutil.rmtree('/content/EDVR/codes/video/inframes/video_subfolders', ignore_errors=True, onerror=None)


def moveProcessedFrames():
  shutil.rmtree(inframes_root)
  os.rename(outframes_root, inframes_root)



def build_video(source_path):
        out_path = result_folder / (
            source_path.name.replace('.mp4', '_no_audio.mp4')
        )
        outframes_folder = outframes_root / (source_path.stem)
        outframes_path_template = str(outframes_folder / '%5d.jpg')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()
        fps = get_fps(source_path)
        print('Original FPS is: ', fps)

        ffmpeg.input(
            str(outframes_path_template),
            format='image2',
            vcodec='mjpeg',
            framerate=fps,
        ).output(str(out_path), crf=17, vcodec='libx264').run(capture_stdout=True)

        result_path = result_folder / source_path.name
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
        print('Video created here: ' + str(result_path))
        return result_path
