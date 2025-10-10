#!/bin/bash
python train.py -t fe -b 32 -m resnet34
python train.py -t fs -b 32 -m resnet34
python train.py -t i2i -b 32 -m resnet34
python train.py -t t2i -b 32 -m resnet34
