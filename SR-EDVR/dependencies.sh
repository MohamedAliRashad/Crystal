#! /bin/bash/
##  cwd = SR-EDVR
cd ./codes/archs/dcn/
python setup.py develop
cd ../../../experiments/pretrained_models/
## get stage 1 models
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1PYULZmtpsmY4Wx8M9f4owdLIwcwQFEmi' -O EDVR_REDS_SR_L.pth
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1ZCl0aU8isEnUCsUYv9rIZZQrGo7vBFUH' -O EDVR_REDS_deblur_L.pth
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1SGVehpZt4WL_X8Jh6blyqmHpc8DdImgv' -O EDVR_REDS_deblurcomp_L.pth
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=18ev7Zx_10-C8-0tAVAe_BpYeLHpr_ChE' -O EDVR_REDS_SRblur_L.pth
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1I7x87ee3E1DoFVgMxX09nfIb2tdUdE3x' -O EDVR_Vimeo90K_SR_L.pth



## get stage 2 models
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1G466gQ1rRl8MUKSEbtaR0U5xgIWdsG66' -O EDVR_REDS_deblurcomp_Stage2.pth
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1kfArevFT8hzbUT2QWXFmUl983LTebQGP' -O EDVR_REDS_SR_Stage2.pth
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=13c-VxMdf8h7MGX-_y4xamxo1hhOMYzsH' -O EDVR_REDS_SRblur_Stage2.pth
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1Y1y6v40dL74Kgf5fxbGd0QC010LFCBYz' -O EDVR_REDS_deblur_Stage2.pth


