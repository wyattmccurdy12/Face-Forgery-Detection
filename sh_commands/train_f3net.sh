#!/bin/bash
python train.py -b 32 -m f3net -t fs
python train.py -b 32 -m f3net -t fe
python train.py -b 32 -m f3net -t i2i
python train.py -b 32 -m f3net -t t2i
