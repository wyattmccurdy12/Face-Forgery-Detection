# FacialForgeryDetection
A repository containing training, validation and testing tools for different facial forgery detection strategies, using the DiFF dataset.


## Instructions
### Train
To train one of the baseline models, run the training script (without validation) `train_no_val.py`.
Use the following command: `python train_no_val.py -t <type of data modification> -b <batch size> -m <model name>`

For example, to train EfficientNet on the FE data modification, with a batch size of 32, use:
`python train_no_val.py -t fe -b 32 -m effnet`
The model will train automatically for 50 epochs, saving a model path every 10 epochs. 

### Test
In order to test the model, run `test_face_net.py`.

The command is as follows: 

`python test_face_net.py -n <network weights path> -b <batch size> -t <network type>`

For example: 
`python test_face_net.py -n final_effnet_fe_E50.pth -b 32 -t effnet`
The script should provide you with an output on the terminal as well as a text file, stored in the `test_results` folder. 

# Other Details
## Where models are defined:
### Xception: 
Imported pretrained xception from timm library. Line 84 of `train_no_val.py`.
### EfficientNet: 
Imported pretrained EfficientNet from timm library. Line 86 of `train_no_val.py`.
### F3-Net
Cloned repository from [https://github.com/yyk-wew/F3Net.git]. What seems like a widely used unofficial implementation. 
Stored in folder `F3Net`, and stored pretrained xception weights in folder `pretrained`. 
The model is called on `train_no_val.py` line 88.
