# Binary Image Classifier and Organize CelebA Dataset!

Hi! In this project, I will guide you to organize CelebA dataset for each attributes and build binary image classifier in PyTorch.


### Steps
I will follow follow 3 steps;
#### 1. Organize Data
#### 2. Train Model
#### 3. Model Usage


## Organize Data
Firstly, we need to download CelebA Align&Cropped Images from [here.](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) Also, we will download the list_attr_celeba.csv file [here](https://www.kaggle.com/jessicali9530/celeba-dataset) to prepare dataset using the features. You need to place both of these folder to root. In csv file, -1 and 1 means that attributes status. In this project, I chose the beard and mustache as an attributes. You can change them in the project you will do yourself. If you want to change, you can modify `OrganizeAttributes.py` file. Then run the file.  Our train dataset ready. We will place both Yes and No folders into one training folder. Then we will create our test data with using `OrganizeTest.py` file. I selected 2000 test data for each Yes and No folder. You can modify your own. Finally our train and test datasets are ready.

## Create Network

Now, we will create our network in `BinaryClassifier.py` file. There are some parameters that you can modify. I used my owns. Also, you can modify network with respect to your project. When you run file, you will see the summary of models. In my computer, I have GTX 1050Ti and training starting in almost 200 seconds.
- Epoch 1/50, Training Loss: 0.417, Training Accuracy: 84.000, Testing Loss: 0.034, Testing Acc: 50.000, Time: 245.3589s
- Epoch 2/50, Training Loss: 0.382, Training Accuracy: 84.000, Testing Loss: 0.023, Testing Acc: 58.000, Time: 231.649s

## Model Usage

After training finished, we will try to use our model for classification. I am going to use MTCNN for detect face and cropping bounding box before insert image into model. In `UseClassifier.py` we insert input image to model for prediction.

## Requirements
- Torch
- Torchvision
- Matplotlib
- Numpy
- MTCNN
- Opencv-Python
- Pillow
- Pandas

> pip install -r requirements.txt

## Final

Your folder should be like image;
![Folder](/imgs/2.png)