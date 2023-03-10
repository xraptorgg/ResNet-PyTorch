# ResNet implementation in PyTorch


This is an implementation of ResNet architecture proposed by Kaiming He et al. in the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) using PyTorch. The files contain implementation of ResNet-34/50/101/152, to instantiate a specific architecture pass in the network depth as argument to the ResNet class. For example, to create a ResNet-50 object with 200 classes, use `ResNet(num_classes = 20, config = 50)`.

The Jupyter Notebook contains details about the architecture and implementation steps, the Python script contains the code.

The Jupyter Notebook and Python files also contain image pipeline for the Tiny ImageNet dataset, however I did not train the model on the dataset due to hardware limitations. If you wish to train the model using the Tiny ImageNet dataset then you should download it from [Tiny-ImageNet-200](http://cs231n.stanford.edu/tiny-imagenet-200.zip), I did not include the dataset in the repository as it is quite large, however it is very straight forward to download and train the model after you download it, just mention the file path of the `tiny-imagenet-200` folder in the `DATA_PATH` in `main.py`.


**NOTE:** This repository contains the model architecture of the original paper as proposed by He et al., the original architecture was trained on the ILSVRC 2015 dataset consisting of 1.2 million images distributed among 1000 classes. While the ResNet architecture is a good model for image classification tasks, the hyperparameters such as number of activation maps, kernel size, stride, etc must be tuned according to the problem.

<div>
<img src="https://cdn.discordapp.com/attachments/418819379174572043/1083694086327894056/ResNet.png" width="1100" alt = "Deep Residual Learning for Image Recognition, Kaiming He et al.">
</div>
