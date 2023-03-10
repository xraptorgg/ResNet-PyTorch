"""
Implementation of original ResNet architecture in PyTorch, from the paper
"Deep Residual Learning for Image Recognition" by Kaiming He et al.
at: https://arxiv.org/pdf/1512.03385.pdf
"""


# importing necessary libraries

import torch
import torch.nn as nn



# ResNet architecture
# supports ResNet-34, 50, 101, 152
# need to pass in the depth as argument


class ResNet(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.block_config = {
            34 : [3, 4, 6, 3],
            50 : [3, 4, 6, 3],
            101 : [3, 4, 23, 3],
            152 : [3, 8, 36, 3]
        }
        self.channel_config = {
            34 : [64, 64, 128, 256, 512],
            50 : [64, 256, 512, 1024, 2048],
            101 : [64, 256, 512, 1024, 2048],
            152 : [64, 256, 512, 1024, 2048]
        }

        self.conv_1  = conv_block(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2, padding = 3)
        self.maxpool_1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.conv_2 = self._make_block(num_blocks = self.block_config[config][0], num = 0, config = config)
        self.conv_3 = self._make_block(num_blocks = self.block_config[config][1], num = 1, config = config)
        self.conv_4 = self._make_block(num_blocks = self.block_config[config][2], num = 2, config = config)
        self.conv_5 = self._make_block(num_blocks = self.block_config[config][3], num = 3, config = config)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features = self.channel_config[config][-1], out_features = num_classes)


        self.init_weights()


    # function to create blocks of residual layers

    def _make_block(self, num_blocks, num, config):
        block = nn.ModuleList([])

        if config == 34:
            block.append(residual_block(in_channels = self.channel_config[config][num], out_channels = self.channel_config[config][num + 1]))
        else:
            block.append(bottleneck_block(in_channels = self.channel_config[config][num], out_channels = self.channel_config[config][num + 1]))

        for i in range(num_blocks - 1):
            if config == 34:
                block.append(residual_block(in_channels = self.channel_config[config][num + 1], out_channels = self.channel_config[config][num + 1]))
            else:
                block.append(bottleneck_block(in_channels = self.channel_config[config][num + 1], out_channels = self.channel_config[config][num + 1]))
        return block


    # kaiming he initialization
    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
    
    
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.maxpool_1(x)

        for block in self.conv_2:
            x = block(x)
        for block in self.conv_3:
            x = block(x)
        for block in self.conv_4:
            x = block(x)
        for block in self.conv_5:
            x = block(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        return self.fc(x)


# bottleneck block used in ResNet-50 and up

class bottleneck_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.downsample = out_channels // in_channels == 2
        self.first = out_channels // in_channels == 4

        stride = 1
        res_channels = in_channels // 4

        if self.downsample:
            res_channels = in_channels // 2
            stride = 2
            self.project = conv_block(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = stride, padding = 0)


        if self.first:
            res_channels = in_channels
            stride = 1
            self.project = conv_block(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = stride, padding = 0)


        self.conv_1 = conv_block(in_channels = in_channels, out_channels = res_channels, kernel_size = 1, stride = 1, padding = 0)
        self.conv_2 = conv_block(in_channels = res_channels, out_channels = res_channels, kernel_size = 3, stride = stride, padding = 1)
        self.conv_3 = conv_block(in_channels = res_channels, out_channels = out_channels, kernel_size = 1, stride = 1, padding = 0)
        self.relu = nn.ReLU()

    def forward(self, x):
        f = self.conv_1(x)
        f = self.relu(f)
        f = self.conv_2(f)
        f = self.relu(f)
        f = self.conv_3(f)
        f = self.relu(f)

        if self.downsample or self.first:
            x = self.project(x)

        h = f + x
        return self.relu(h)



# simple residual blocks used in ResNet-34 and below

class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.downsample = in_channels != out_channels

        stride = 1

        if self.downsample:
            stride = 2
            self.project = conv_block(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = stride, padding = 0)

        self.conv_1 = conv_block(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.conv_2 = conv_block(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        f = self.conv_1(x)
        f = self.relu(f)
        f = self.conv_2(f)
        f = self.relu(f)

        if self.downsample:
            x = self.project(x)

        h = f + x
        return self.relu(h)


# convlutional layer and batch norm block

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        return self.bn(x)