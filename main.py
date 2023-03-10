import torch
import torch.nn as nn
import tiny_imagenet_data as dt
import model_functions as mf
import resnet_model as res


# hyperparameters

DATA_PATH = "resnet/ResNet-PyTorch/tiny-imagenet-200"
BATCH_SIZE = 32
CLASSES = 200
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
FACTOR = 0.1
EPOCHS = 600000


# device agnostic code

device = "cuda" if torch.cuda.is_available() else "cpu"

# train and test metrics 

TRAIN_LOSS_LIST = []
VAL_LOSS_LIST = []
VAL_ACC_LIST = []
EPOCH_COUNT_LIST = []




if __name__ == "__main__":
    
    # create data

    train_data, val_data = dt.prepare_dataset(data_path = DATA_PATH)

    train_dataloader, val_dataloader = dt.prepare_dataloader(batch_size = BATCH_SIZE, training_data = train_data, val_data = val_data)

    # instantiating models

    resnet_50 = res.ResNet(num_classes = CLASSES, config = 50).to(device)

    # loss function and optimizer

    cross_entropy = nn.CrossEntropyLoss()
    sgd = torch.optim.SGD(params = resnet_50.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)
    learning_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = sgd, mode = "min", factor = FACTOR)


    # model training

    torch.manual_seed(1234)
    mf.model_train(device = device, epochs = EPOCHS, model = resnet_50, train_dataloader = train_dataloader, val_dataloader = val_dataloader, 
                    loss_func = cross_entropy, optimizer = sgd, scheduler = learning_decay, epoch_count = EPOCH_COUNT_LIST, 
                    train_loss_values = TRAIN_LOSS_LIST, val_loss_values = VAL_LOSS_LIST,
                    val_acc_values = VAL_ACC_LIST)