# Facial Forgery Detection 
A repository containing training, validation and testing tools for different facial forgery detection strategies, using the DiFF dataset.

## Instructions

### Build Subsets
Run all cells in `Subset_Train_Test_Data.ipynb`. This should provide you with the subsetted version of the data used in our experiments, provided you have downloaded the DIFF dataset (https://github.com/xaCheng1996/DiFF) into your train and test folders (we used the following directory structure): 

./train
   - /fe
   - /fs
   - /i2i
   - /t2i
   - /real

./val
   - /fe
   - /fs
   - /i2i
   - /t2i
   - /real

./test
   - /fe
   - /fs
   - /i2i
   - /t2i
   - /real

### Train
To train one of the baseline models, run the training script (without validation) `train_no_val.py`.
Use the following command: `python train.py -t <type of data modification> -b <batch size> -m <model name>`

For example, to train EfficientNet on the FE data modification, with a batch size of 32, use:
`python train.py -t fe -b 32 -m effnet`
The model will train automatically for 50 epochs, saving the best-performing model as it goes.

### Test
In order to test the model, run `test.py`.

The command is as follows: 

`python test.py -n <network weights path> -b <batch size> -t <network type>`

For example: 
`python test.py -n final_effnet_fe_E50.pth -b 32 -t effnet`
The script should provide you with an output on the terminal as well as a text file, stored in the `test_results` folder. 

Note: 
Bash scripts for training runs of all baseline models exist in `sh_commands` folder.

## Acknowledgements
We thank the authors of the following paper for the free availability of their high-quality dataset: 

`
@misc{cheng2024diffusion,
      title={Diffusion Facial Forgery Detection}, 
      author={Harry Cheng and Yangyang Guo and Tianyi Wang and Liqiang Nie and Mohan Kankanhalli},
      year={2024},
      eprint={2401.15859},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
`
