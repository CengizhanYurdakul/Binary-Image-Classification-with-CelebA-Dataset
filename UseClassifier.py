import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3,stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(8)        
        self.relu = nn.ReLU()                 
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)   
        
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)    
        
        self.fc1 = nn.Linear(in_features=8192, out_features=4000)   
        self.droput = nn.Dropout(p=0.3)                    
        self.fc2 = nn.Linear(in_features=4000, out_features=2000)
        self.droput = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(in_features=2000, out_features=500)
        self.droput = nn.Dropout(p=0.3)
        self.fc4 = nn.Linear(in_features=500, out_features=50)
        self.droput = nn.Dropout(p=0.3)
        self.fc5 = nn.Linear(in_features=50, out_features=2)    
       
        
    def forward(self,x):
        out = self.cnn1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        out = out.view(-1,8192)   

        out = self.fc1(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc5(out)
        return out

def predict(image,model):
    transform_ori = transforms.Compose([transforms.RandomResizedCrop(64),  
                                    transforms.RandomHorizontalFlip(), 
                                    transforms.ToTensor(),              
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img = Image.fromarray(image)  
    img = transform_ori(img)    
    img = img.view(1,3,64,64)     
    img = Variable(img)      

    
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        img = img.cuda()

    output = model(img)
    print(output)
    print(output.data)
    _, predicted = torch.max(output,1)
    if predicted.item()==0:
        p = 'No Beard'
    else:
        p = 'Yes Beard'
    return  p




model = CNN()
model.load_state_dict(torch.load("./models/Beard31.pth")) # Load model that you trained
model.eval()

detector = MTCNN()
img = cv2.imread("./imgs/withoutbeard.jpg") # Input Image
result = detector.detect_faces(img)
x,y,w,h = result[0]["box"]
cropped = img[y:y+h, x:x+w]


predicted = predict(cropped, model)
print(predicted)