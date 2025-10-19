import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class DQN(nn.Module):
    def __init__(self, n_frames=4, n_actions=3):
        super(DQN, self).__init__()
        
        self.conv1 = nn.Conv2d(n_frames, 64, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.res_block1 = ResidualBlock(256)
        self.res_block2 = ResidualBlock(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.res_block3 = ResidualBlock(512)
        self.res_block4 = ResidualBlock(512)
        
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(1024)
        
        self.res_block5 = ResidualBlock(1024)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        
        x = self.res_block5(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
