<p align="center">
  <img width="323" height="200" src="https://github.com/MohamedAliRashad/Crystal/blob/master/static/assets/svg/icon.svg">
</p>

# Crystal
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohamedAliRashad/Crystal/blob/master/Crystal.ipynb)
[![YouTube Video Views](https://img.shields.io/youtube/views/0lbod6pHAwg?label=YouTube&logo=YouTube&logoColor=Red&style=plastic)](https://www.youtube.com/watch?v=0lbod6pHAwg)

Crystalize What's Important in Life

## How To Use

1. Download all required dependencies with this
```pip3 install -r requirements.txt```.

2. Run the app with this ```python3 app.py```

3. Go to this http://0.0.0.0:5000/

**Note**: [**Cuda**](https://developer.nvidia.com/cuda-downloads) and [**Cudnn**](https://developer.nvidia.com/cuda-downloads) is required for this to work and we don't provide them (so you are on your own).

**Note**: Open an issue with any problem you face, we might help :cowboy_hat_face:

## News
- **2 June 2020** Added Few Examples to show results of our pipline [YouTube](https://www.youtube.com/watch?v=0lbod6pHAwg)
- **1 June 2020** Made a [YouTube](https://www.youtube.com/watch?v=Fuum9eexgUg) video demonstrating what we have accomplished so far 
- **29 May 2020** Tested Colab Environment & Seperation of the pipeline in `main.py`
- **14 May 2020** Added New Icon & integration of a better Super-Resolution techniques

## Future Work
- [x] Provide a working colab notebook in case there was problems in getting the site up.
- [ ] Consider making an API for developers.
- [ ] Add YouTube video downloading feaure.
- [ ] Add content in tour, explore .. pages.
- [x] Optimize the code for faster inference.
- [x] Make a Dockerfile of the project for easy deployment.
- [ ] Deploy on a server with a good domain name.

## :sparkles: Huge thanks for the real heroes [Here](https://github.com/baowenbo/DAIN) and [Here](https://github.com/xinntao/EDVR):sparkles:
If you find this work or code useful for your research, please cite:
```
@inproceedings{DAIN,
    author    = {Bao, Wenbo and Lai, Wei-Sheng and Ma, Chao and Zhang, Xiaoyun and Gao, Zhiyong and Yang, Ming-Hsuan}, 
    title     = {Depth-Aware Video Frame Interpolation}, 
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
    year      = {2019}
}
@article{MEMC-Net,
     title={MEMC-Net: Motion Estimation and Motion Compensation Driven Neural Network for Video Interpolation and Enhancement},
     author={Bao, Wenbo and Lai, Wei-Sheng, and Zhang, Xiaoyun and Gao, Zhiyong and Yang, Ming-Hsuan},
     journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
     doi={10.1109/TPAMI.2019.2941941},
     year={2018}
}
@InProceedings{wang2019edvr,
  author    = {Wang, Xintao and Chan, Kelvin C.K. and Yu, Ke and Dong, Chao and Loy, Chen Change},
  title     = {EDVR: Video restoration with enhanced deformable convolutional networks},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  month     = {June},
  year      = {2019},
}
@Article{tian2018tdan,
  author    = {Tian, Yapeng and Zhang, Yulun and Fu, Yun and Xu, Chenliang},
  title     = {TDAN: Temporally deformable alignment network for video super-resolution},
  journal   = {arXiv preprint arXiv:1812.02898},
  year      = {2018},
}
```
