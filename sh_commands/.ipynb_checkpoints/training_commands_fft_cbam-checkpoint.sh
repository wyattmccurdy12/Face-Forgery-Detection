#!/bin/bash
python train_no_val.py -t fe -b 32 -m resnet_cbam_i -fft Y
python train_no_val.py -t fs -b 32 -m resnet_cbam_i -fft Y
python train_no_val.py -t i2i -b 32 -m resnet_cbam_i -fft Y
python train_no_val.py -t t2i -b 32 -m resnet_cbam_i -fft Y