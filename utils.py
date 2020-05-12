import imageio
import tqdm
import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import numpy as np
import torch

def video2frames(video_path, ):

    reader = imageio.get_reader(video_path)
    meta_data = reader.get_meta_data()
    fps = meta_data['fps']
    size = meta_data['size']
    n_frames = meta_data['nframes']

    for i, img in tqdm.tqdm(enumerate(reader), total=n_frames):
        imageio.imsave(os.path.join(FRAMES_FOLDER, '%08d.jpg' % i), img)

    return {"fps": fps, "name":video_path.split("/")[-1].split(".")[0], "size":size}

def frames2video(frames_path, meta_data):

    os.system("ffmpeg -framerate {} -i %06d.jpg {}.mp4".format(meta_data["fps"], meta_data["name"]))

def load_DAIN():
    # Let the magic happen
    from DAIN.DAIN import DAIN
    module = DAIN()
    
    # load the weights online
    from torch.hub import load_state_dict_from_url
    state_dict = load_state_dict_from_url("http://vllab1.ucmerced.edu/~wenbobao/DAIN/best.pth")
    module.load_state_dict(state_dict)

    return module

def infer_DAIN(model, meta_data):

    model.cuda()
    frames = sorted(os.listdir(FRAMES_FOLDER))
    for frame in frames:
        os.rename(frame, str(int(frame.split(".")[0])*2).zfill(6))
    
    frames = sorted(os.listdir(FRAMES_FOLDER))
    scale_precent = 50
    width = int(meta_data["size"][0] * scale_precent / 100)
    height = int(meta_data["size"][1] * scale_precent / 100)
    dim = (width, height)
    model.eval()
    for i in tqdm(range(len(frames) - 1)):

        image1 = cv2.resize(imageio.imread(frames[i]), dim, interpolation=cv2.INTER_AREA)
        image2 = cv2.resize(imageio.imread(frames[i + 1]), dim, interpolation=cv2.INTER_AREA)
        
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

        imageio.imsave(os.path.join(FRAMES_FOLDER, str(2*i + 1).zfill(6) + ".jpg"), cv2.resize(np.round(y_).astype(np.uint8), meta_data["size"], interpolation=cv2.INTER_AREA))

    meta_data["fps"] = meta_data["fps"]*2

    return meta_data
