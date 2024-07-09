import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from bc_audio.utils import FEATURE_SETS, extract_raw_data, process_data, DIR_PATH
from pathlib import Path
from bc_video.lstm_utils import save_model
import random
import re

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns


class Classifier(nn.Module):
    def __init__(self, number_of_features):
        super().__init__()
        self.layer1 = nn.Linear(number_of_features, 5)                        
        self.layer2 = nn.Linear(5, 5)
        self.layer3 = nn.Linear(5, 1)

        self.bc10 = nn.BatchNorm1d(5)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(p=0.2)

        nn.init.xavier_normal_(self.layer1.weight) 
        nn.init.xavier_normal_(self.layer2.weight)
        nn.init.xavier_normal_(self.layer3.weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bc10(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.bc10(x)
        x = self.relu(x)
        #x = self.dropout(x)        
        x = self.layer3(x)
        
        return x

def predict(lr, ep, batch_len, wd, model, x_train, x_test, y_train, y_test, experiment_number, run):
    loss_function = nn.BCEWithLogitsLoss()
    best_score = 0
    learning_rate = lr  # 0.0005
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay= wd) #optim.SGD(model.parameters(), lr=learning_rate)
    df_train = {"Loss": [], "Accuracy": []}
    df_test = {"Loss": [], "Accuracy": []}
    model = model.to(device)
    epochs = ep  # 100
    batch_size = batch_len  # 256
    batch_start_train = torch.arange(0, len(x_train), batch_size)
    batch_start_test = torch.arange(0, len(x_test), batch_size)
    ctp_df_merged = pd.DataFrame()
    for epoch in range(epochs):
        print(f'\n Epoch: {epoch}')
        correct_train_preds = 0
        total_tr = 0
        train_loss_list = []
        model.train()
        with tqdm.tqdm(
            batch_start_train, 
            unit="batch", 
            mininterval=0, 
            disable=True, 
            desc="Train"
        ) as bar:
            bar.set_description(f"Train Epoch {epoch}")
            for start in bar:
                optimizer.zero_grad()

                x_batch_train = x_train[start : start + batch_size].to(device)
                y_batch_train = y_train[start : start + batch_size].to(device)
                #train_pred_raw = model(x_batch_train)  # forward pass .float()

                train_pred = model(x_batch_train)

                train_loss = loss_function(train_pred, y_batch_train)

                train_pred = train_pred.clamp(0,1)

                #_, train_pred = torch.max(train_pred.data, 1)
                
                train_loss_list.append(float(train_loss))  

                train_loss.backward()  # backward pass
                
                optimizer.step()  # update weights

                total_tr += y_batch_train.size(0)
                correct_train_preds += (train_pred.reshape(-1,1) == y_batch_train).sum().item() #correct_train_preds += (train_pred.round() == y_batch_train).sum().item()

        df_train["Loss"].append(np.mean(train_loss_list))
        df_train["Accuracy"].append(100 * correct_train_preds / total_tr)
        print(total_tr, correct_train_preds, correct_train_preds/total_tr, y_batch_train.shape)
        print(
            "Train Epoch [{}/{}], Accuracy: {:0.4f}; Loss: {:.4f}".format(
                epoch + 1,
                epochs,
                100 * correct_train_preds / total_tr,
                np.mean(train_loss_list),
            )
        )
        with torch.no_grad():
            model.eval()
            correct_test_preds = 0
            total_te = 0
            test_loss_list = []
            with tqdm.tqdm(
                batch_start_test,
                unit="batch",
                mininterval=0,
                disable=True,
                desc="Test"
            ) as bar1:
                bar1.set_description(f"Test Epoch {epoch}")
                for start in bar1:
                    x_batch_test = x_test[start : start + batch_size].to(device)
                    y_batch_test = y_test[start : start + batch_size].to(device)
                    #test_pred_raw = model(x_batch_test)
                    test_pred = model(x_batch_test)
                    total_te += y_batch_test.size(0)

                    test_loss = loss_function(test_pred, y_batch_test)
                    
                    test_pred = test_pred.clamp(0,1)
                    test_loss_list.append(float(test_loss))

                    #print('before: ', test_pred[0])
                    
                    #_, test_pred = torch.max(test_pred.data, 1)
                    
                    #print('after: ',test_pred[0])
                    #print(test_pred.shape, y_batch_test.shape)
                    correct_test_preds += (test_pred.reshape(-1,1) == y_batch_test).sum().item() #correct_test_preds += (test_pred.round() == y_batch_test).sum().item()
                    #print('y_data: ', y_batch_test[0])
                    #print(test_pred.reshape(-1,1).shape, y_batch_test.shape)
                    ctp_df = pd.DataFrame(test_pred.cpu().numpy())
                    ctp_df_merged = pd.concat([ctp_df_merged, ctp_df])
        
            df_test["Loss"].append(np.mean(test_loss_list))
            df_test["Accuracy"].append(100 * correct_test_preds / total_te)
            print(
                "Test Epoch [{}/{}], Accuracy: {:0.4f}; Loss: {:.4f}, CTP: {}".format(
                    epoch + 1,
                    epochs,
                    100 * correct_test_preds / total_te,
                    np.mean(test_loss_list),
                    correct_test_preds
                )
            )

            # print(ctp_df_merged.value_counts())
        if (100 * correct_test_preds / (len(x_test))) > best_score:
            best_score = 100 * correct_test_preds / (len(x_test))
            print('New best test metric observed! Saving Model..')

            if not os.path.exists(DIR_PATH / f'inference/mlp/experiment-{experiment_number}/models/run-{run}/'):
                os.makedirs(DIR_PATH / f'inference/mlp/experiment-{experiment_number}/models/run-{run}/')
                
            model_savepath = DIR_PATH / f'inference/mlp/experiment-{experiment_number}/models/run-{run}/'
            save_model(model,epoch,optimizer,model_savepath )

    return df_train, df_test, ctp_df_merged

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

if __name__ == "_main_":

    feature_set_name = 'eGe' #'GeM', 'eGe', 

    extractor = FEATURE_SETS[feature_set_name]['extractor']

    # features = [ 'pcm_zcr_sma',
    #             'F0env_sma',
    #             #'pcm_loudness_sma',
    #             ] #Custom features

    # features = [    'mfcc_sma[1]',
    #                 'mfcc_sma[2]',
    #                 'mfcc_sma[3]',
    #                 'mfcc_sma[4]',
    #                 'mfcc_sma[5]',
    #                 'mfcc_sma[6]',
    #                 'mfcc_sma[7]',
    #                 'mfcc_sma[8]',
    #                 'mfcc_sma[9]',
    #                 'mfcc_sma[10]',
    #                 'mfcc_sma[11]',
    #                 'mfcc_sma[12]'
    #             ]
    features = [ 
                    'F0semitoneFrom27.5Hz_sma3nz',
                ]
    
    aud_mean_data = extract_raw_data(extractor, feature_set_name, features, custom_window_length_factor = 4)

    train_test_split_ratio = 0.7
    
    X_train, X_test, Y_train, Y_test, stats = process_data(train_test_split_ratio, aud_mean_data, features)

    for key in stats:
        print(key,':',stats[key])

    if torch.cuda.is_available():

        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    num_of_features = X_train.shape[1]
    Model = Classifier(num_of_features)
    Hyperparams = {
        "Learning rate": [ 0.001], # , 0.05, 0.001, 0.005 , 0.001, 0.0001, 0.0005, 0.00001, 0.00005, 0.000001
        "Epochs": [100], #,, 100, 200 , 250, 500, 750
        "Batch size": [512, 1024, 2048], # , 64, 128 , 32, 64, 128
        "Weight decay": [1e-4]
    }
    experiment = 8
    csv_save_path = DIR_PATH / f'inference/mlp/experiment-{experiment}/data'

    if not os.path.exists(csv_save_path):
        os.makedirs(csv_save_path)

    results = []
    run = 0
    for lr in Hyperparams["Learning rate"]:
        for ep in Hyperparams["Epochs"]:
            for bs in Hyperparams["Batch size"]:
                for w in Hyperparams["Weight decay"]:
                    dataframe_train, dataframe_test, cdf_m = predict(
                        lr, ep, bs, w, Model, X_train, X_test, Y_train, Y_test, experiment, run
                    )
                    results.append(
                        {
                            "Iteration": run,
                            "Learning rate": lr,
                            "Epoch": ep,
                            "Batch size": bs,
                            "Train acc": dataframe_train["Accuracy"],
                            "Train loss": dataframe_train["Loss"],
                            "Test acc": dataframe_test["Accuracy"],
                            "Test loss": dataframe_test["Loss"],
                            "Final Train acc": dataframe_train["Accuracy"][-1],
                            "Final Test acc": dataframe_test["Accuracy"][-1],
                        }
                    )
                    csv_filename = csv_save_path / f'Run - {run}_{lr}_{ep}_{bs}.csv'
                    results_df=pd.DataFrame()
                    results_df['Train acc'] = dataframe_train["Accuracy"]
                    results_df['Test acc'] = dataframe_test["Accuracy"]
                    results_df['Train loss'] = dataframe_train["Loss"]
                    results_df['Test loss'] = dataframe_test["Loss"]
                    results_df.to_csv(csv_filename, index=False)
                    run += 1
                    print("Run:{0} ; Train acc: {1} ; Test acc: {2} for ( {3},{4},{5} )".format(
                        run,
                        dataframe_train["Accuracy"][-1],
                        dataframe_test["Accuracy"][-1],
                        lr,
                        ep,
                        bs
                    ))

    data_path = DIR_PATH / f'inference/mlp/experiment-{experiment}/data/'
    plot_path = DIR_PATH / f'inference/mlp/experiment-{experiment}/plots/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    test_acc_df = pd.DataFrame(columns=['Run', 'Final test acc'])
    csvs = [i.name for i in Path(data_path).glob('*.csv')]
    csvs = sorted_nicely(csvs)
    max_acc = 0.0
    max_acc_run = str()
    for k in csvs:
        #if results.index(k) == 27:
        fig,axs=plt.subplots(2,2)
        fig.suptitle(k[:-4])
        fig.tight_layout(pad=2.0)
        dk = pd.read_csv(data_path / k)
        axs[0, 0].plot(dk["Train loss"].astype(float).values)
        axs[0, 0].set_title('Train loss')
        axs[0, 0].set(ylabel='Loss')
        axs[0, 1].plot(dk["Train acc"], 'tab:red')
        axs[0, 1].set_title('Train acc')
        axs[1, 0].plot(dk["Test loss"].astype(float).values, 'tab:green')
        test_acc = dk["Test acc"].values[-1]
        run = k[:-4]
        test_acc_df.loc[len(test_acc_df)] = [run, test_acc]

        if test_acc > max_acc:
            max_acc = test_acc
            max_acc_run = run
        axs[1, 0].set_title('Test loss')
        axs[1, 0].set(xlabel='Epochs', ylabel='Loss')
        axs[1, 1].plot(dk["Test acc"], 'tab:orange')
        axs[1, 1].set_title('Test acc')
        axs[1, 1].set(xlabel='Epochs')
        plt.savefig(plot_path / f'{k[:-4]}.png')
        plt.close()