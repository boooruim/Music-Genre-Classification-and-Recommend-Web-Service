import torch
import torch.nn as nn
# Make a CNN & train it to predict genres.
class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, ksize=7, dropout=0.1):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, ksize, padding = 1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        return out
    

class CNN(nn.Module):
    def __init__(self, num_channels=8, 
                       num_classes=10):
        super(CNN, self).__init__()

        # convolutional layers
        self.layer1 = Conv_2d(3, num_channels)
        self.layer2 = Conv_2d(num_channels, num_channels*2)
        self.layer3 = Conv_2d(num_channels*2, num_channels * 4)
        self.layer4 = Conv_2d(num_channels * 4, num_channels * 8)
        self.layer5 = Conv_2d(num_channels * 8, num_channels * 16,ksize=5)

        # dense layers
        self.dense1 = nn.Linear(num_channels * 16, 256)
        self.dense_bn = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # input Preprocessing
        # print(x.shape) # 
        # convolutional layers
        out = self.layer1(x)
        # print(out.shape) # 
        out = self.layer2(out)
        # print(out.shape) # 

        out = self.layer3(out)
        # print(out.shape) # 

        out = self.layer4(out)
        # print(out.shape) #

        out = self.layer5(out)
        # print(out.shape) # (16,128,1,1)
        # reshape. (batch_size, num_channels, 1, 1) -> (batch_size, num_channels)
        out = out.squeeze()
        # print(out.shape) # (16,128,1,1)

        # dense layers
        out = self.dense1(out)
        out = self.dense_bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.dense2(out)

        return out