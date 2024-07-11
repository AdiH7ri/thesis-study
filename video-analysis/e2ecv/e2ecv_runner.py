import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
from torchvision.transforms import v2


import matplotlib.pyplot as plt
from e2ecv.utils import VidData3D, return_stat_dict, SAVEPATH, DIR_PATH
from e2ecv.models import CNN3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle


loss_function = nn.BCEWithLogitsLoss()

def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    model.train()
    
    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    y_count = 0

    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device) #.view(-1, )

        N_count += X.size(0)
        y_count += y.shape[0]
        optimizer.zero_grad()
        output = model(X)  # output size = (batch, number of classes)

        loss = loss_function(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = output.clamp(0,1) # y_pred != output
        step_score = (y_pred.reshape(-1,1) == y).sum().item() #accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score / y.shape[0]))
    
    train_score = 100* np.sum(scores)/y_count
    train_loss = np.mean(losses)

    print('\nTrain set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(y_count, np.mean(losses) , 100* np.sum(scores)/y_count))
    
    return train_loss, train_score


def validation(model, device, test_loader):
    # set model as testing mode
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device)  #.view(-1, )

            output = model(X)
           
            loss = loss_function(output, y) #F.binary_cross_entropy_with_logits(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.clamp(0,1) # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # to compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = (all_y_pred.reshape(-1,1) == all_y).sum().item() #accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
    test_acc = 100* test_score/all_y.shape[0]
    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, test_acc))

    return test_loss, test_acc, model



def main():

    # training parameters

    params = [
        {
           'epoch':50, 'bs': 4, 'lr': 1e-3, 'dropout': 0.5, 'fc1': 64, 'fc2': 32

        }   
    ]


    EXPERIMENT = 32

    for param in params:

        # set savepaths    

        DATAPATH = SAVEPATH / 'size-256x256' 

        EPOCH_DATA_SAVEPATH = DIR_PATH / 'inference' / f'experiment-{EXPERIMENT}' / 'data'

        EPOCH_PLOT_SAVEPATH = DIR_PATH / 'inference' / f'experiment-{EXPERIMENT}' / 'plots'

        MODEL_SAVEPATH = DIR_PATH / 'inference' / f'experiment-{EXPERIMENT}' / 'models'

        OPTIM_SAVEPATH = DIR_PATH / 'inference' / f'experiment-{EXPERIMENT}' / 'optim'

        epochs = param['epoch']
        batch_size = param['bs']
        learning_rate = param['lr']
        log_interval = 1
        img_x, img_y = 224, 224  # resize video 2d frame size

        # 3D CNN parameters
        WINDOW_LENGTH = 15*25
        fc_hidden1, fc_hidden2 = param['fc1'], param['fc2']
        dropout = param['dropout']        # dropout probability

        # Detect devices
        use_cuda = torch.cuda.is_available()                   # check if GPU exists
        device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

        # load UCF101 actions names
        params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 8, 'pin_memory': True} if use_cuda else {}

        train_pids = ['N252', 'P265', 'N254', 'N257', 'P314', 'P243', 'N334', 'N255', 'P317', 'P244', 'N260', 'N249', 'P240', 'P263', 'N335', 'P237', 'N261', 'N332', 'N258', 'P203', 'P316', 'P246', 'N251', 'P245']
        test_pids = ['P239', 'N250', 'P318', 'P264', 'P315', 'N253', 'N256', 'P242', 'P248', 'N333', 'N262']

        tr_dict = return_stat_dict(DATAPATH, train_pids, WINDOW_LENGTH)
        te_dict = return_stat_dict(DATAPATH, test_pids, WINDOW_LENGTH)


        # img_transforms = transforms.Compose([ 
        #     transforms.Resize([img_x, img_y]),
        #     transforms.ToTensor(),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.Normalize(mean=[0.5], std=[0.5])
        # ]) 

        train_img_transforms = v2.Compose([
                                v2.Resize([img_x, img_y]),
                                v2.RandomRotation(degrees=(-5, 5), interpolation=cv2.INTER_CUBIC),
                                #v2.RandomCrop(height=img_x, width=img_y),
                                v2.RandomHorizontalFlip(p=0.2),
                                v2.ColorJitter(brightness=(1,5)), 
                                #v2.RandomInvert(p = 0.5),
                                v2.PILToTensor(),
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(mean=[0.5], std=[0.5]),
                                ])

        test_img_transforms =v2.Compose([
                                v2.Resize([img_x, img_y]),
                                #v2.RandomCrop(height=img_x, width=img_y),
                                v2.RandomHorizontalFlip(p=0.2),
                                v2.PILToTensor(),
                                v2.ToDtype(torch.float32, scale=True),
                                v2.Normalize(mean=[0.5], std=[0.5]),
                                ])


        train_set, valid_set = VidData3D(data_file=tr_dict,  transform=train_img_transforms), \
                            VidData3D(data_file=te_dict, transform=test_img_transforms)

        train_loader = data.DataLoader(train_set, **params)
        valid_loader = data.DataLoader(valid_set, **params)

        print('Train & test data loaded successfully! \n')
        # create model
        cnn3d = CNN3D(t_dim=WINDOW_LENGTH, img_x=img_x, img_y=img_y,
                    drop_p=dropout, fc_hidden1=fc_hidden1,  fc_hidden2=fc_hidden2).to(device)


        optimizer = torch.optim.Adam(cnn3d.parameters(), lr=learning_rate)   # optimize all cnn parameters


        # record training process
        epoch_train_losses = []
        epoch_train_scores = []
        epoch_test_losses = []
        epoch_test_scores = []

        best_test_score = 0
        # start training
        print('Beginning training.... \n')
        for epoch in range(epochs):
            # train, test model
            train_losses, train_scores = train(log_interval, cnn3d, device, train_loader, optimizer, epoch)
            epoch_test_loss, epoch_test_score, cnn3d = validation(cnn3d, device, valid_loader)

            # save Pytorch models of best record
            if not MODEL_SAVEPATH.exists():
                os.makedirs(MODEL_SAVEPATH)
            
            if not OPTIM_SAVEPATH.exists():
                os.makedirs(OPTIM_SAVEPATH)

            if epoch_test_score > best_test_score:
                
                best_test_score = epoch_test_score
                
                torch.save(cnn3d.state_dict(),  MODEL_SAVEPATH / f'3dcnn_best_model.pth')  # save spatial_encoder

                torch.save(optimizer.state_dict(), OPTIM_SAVEPATH / f'3dcnn_best_model_optimizer.pth')      # save optimizer
            
                print("Epoch {} model saved!".format(epoch + 1))

            # save results
            epoch_train_losses.append(train_losses)
            epoch_train_scores.append(train_scores)
            epoch_test_losses.append(epoch_test_loss)
            epoch_test_scores.append(epoch_test_score)

            # save all train test results
            A = np.array(epoch_train_losses)
            B = np.array(epoch_train_scores)
            C = np.array(epoch_test_losses)
            D = np.array(epoch_test_scores)

        if not EPOCH_DATA_SAVEPATH.exists():
            os.makedirs(EPOCH_DATA_SAVEPATH)

        np.save(EPOCH_DATA_SAVEPATH / 'epoch_training_losses.npy', A)
        np.save(EPOCH_DATA_SAVEPATH / 'epoch_training_scores.npy', B)
        np.save(EPOCH_DATA_SAVEPATH / 'epoch_test_loss.npy', C)
        np.save(EPOCH_DATA_SAVEPATH / 'epoch_test_score.npy', D)

        # plot
        fig = plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.plot(np.arange(1, epochs + 1), A)  # train loss (on epoch end) [:, -1]
        plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
        plt.title("Train vs Test Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Test'], loc="upper left")
        # 2nd figure
        plt.subplot(122)
        plt.plot(np.arange(1, epochs + 1), B)  # train accuracy (on epoch end)
        plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
        # plt.plot(histories.losses_val)
        plt.title("Train vs Test Accuracy metric")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Test'], loc="upper left")

        plot_savepath = EPOCH_PLOT_SAVEPATH /"train_test_performance.png"

        if not EPOCH_PLOT_SAVEPATH.exists():
            os.makedirs(EPOCH_PLOT_SAVEPATH)

        plt.savefig(plot_savepath)
        plt.close(fig)
        #plt.show()

        EXPERIMENT += 1

if __name__ == '__main__':
    main()