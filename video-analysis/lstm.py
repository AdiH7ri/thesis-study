import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import tqdm
import logging
import datetime
from matplotlib import pyplot as plt
from pathlib import Path

from bc_video import lstm_utils
from bc_audio.utils import generate_train_val_split, get_metrics

DIR_PATH = Path(__file__).parent


FPS=25
WINDOW_LENGTH=FPS*15
K = 5
train_test_split_ratio = 0.70981

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def get_clean_fold_train_pair_ids(pid_list, test, v_pids):
    t_pid = []
    for i in v_pids:
        clean = [x for x in pid_list if x not in (test + i)]
        t_pid.append(clean)
    return t_pid

def save_plots(s, s1, s2, s3, RB, epochs, sp):
    if s == 'Accuracy':
    
            fig = plt.figure(figsize=(10, 10))
            plt.plot(np.arange(1, epochs + 1), s1)          # train accuracy (on epoch end)
            plt.plot(np.arange(1, epochs + 1), s2)         #  val accuracy (on epoch end)
            plt.plot(np.arange(1, epochs + 1), s3)         #  val accuracy (on epoch end)
            plt.hlines(y=RB['Train'], xmin=0, xmax=epochs, linestyle="dashed", colors='pink')   
            plt.hlines(y=RB['Val'], xmin=0, xmax=epochs, linestyle="dashed", colors='olive')   
            plt.hlines(y=RB['Test'], xmin=0, xmax=epochs, linestyle="dashed", colors='purple')
            plt.vlines(x=s3.index(np.max(s3, axis=0)), ymin=np.min(s1, axis=0), ymax=np.max(s1, axis=0), linestyle="dashed", colors='darkorange')   
            plt.title(f"Obsserved {s}-metric values across Train, Validation and Test splits")
            plt.xlabel('Epochs')
            plt.ylabel(f'{s}')
            plt.legend([f'Train {s}', f'Validation {s}', f'Test {s}', 'Train RB', 'Validation RB', 'Test RB', 'Best Test Epoch'], loc="upper left")

            save_path = sp / f'Train_Val_Test_metric-{s}.png'

            plt.savefig(save_path)
            plt.close(fig)
    else:
            fig = plt.figure(figsize=(10, 10))
            plt.plot(np.arange(1, epochs + 1), s1)          # train accuracy (on epoch end)
            plt.plot(np.arange(1, epochs + 1), s2)         #  val accuracy (on epoch end)
            plt.plot(np.arange(1, epochs + 1), s3)         #  val accuracy (on epoch end)
            plt.title(f"Obsserved {s}-metric values across Train, Validation and Test splits")
            plt.xlabel('Epochs')
            plt.ylabel(f'{s}')
            plt.legend([f'Train {s}', f'Validation {s}', f'Test {s}'], loc="upper left")

            save_path = sp / f'Train_Val_Test_metric-{s}.png'

            plt.savefig(save_path)
            plt.close(fig)

def save_metrics(split, acc, prec, recall, f1, c_k, savepath):
    np.save(savepath / f'{split}_accuracy.npy', acc)
    np.save(savepath / f'{split}_precision.npy', prec)
    np.save(savepath / f'{split}_recall.npy', recall)
    np.save(savepath / f'{split}_f1_score.npy', f1)
    np.save(savepath / f'{split}_cohen_kappa.npy', c_k)

def extract_and_save__metrics_and_plots(tr, val, te, epochs, RB, savepath):
    acc_tr, prec_tr, recall_tr, f1_tr, c_k_tr = [], [], [], [], []
    acc_val, prec_val, recall_val, f1_val, c_k_val = [], [], [], [], []
    acc_te, prec_te, recall_te, f1_te, c_k_te = [], [], [], [], [] 

    data_savepath = savepath / 'data'
    plot_savepath = savepath / 'plots'

    for i in range(epochs):
        
        a_tr,p_tr,r_tr,f_tr,c_tr = get_metrics(np.array(tr[f'{i}_actual']).reshape(-1,1), np.array(tr[f'{i}_predicted']).reshape(-1,1))
        a_vl,p_vl,r_vl,f_vl,c_vl = get_metrics(np.array(val[f'{i}_actual']).reshape(-1,1), np.array(val[f'{i}_predicted']).reshape(-1,1))
        a_te,p_te,r_te,f_te,c_te = get_metrics(np.array(te[f'{i}_actual']).reshape(-1,1), np.array(te[f'{i}_predicted']).reshape(-1,1))

        acc_tr.append(a_tr),prec_tr.append(p_tr),recall_tr.append(r_tr),f1_tr.append(f_tr),c_k_tr.append(c_tr)
        acc_val.append(a_vl),prec_val.append(p_vl),recall_val.append(r_vl),f1_val.append(f_vl),c_k_val.append(c_vl)
        acc_te.append(a_te),prec_te.append(p_te),recall_te.append(r_te),f1_te.append(f_te),c_k_te.append(c_te)

    save_metrics('Train',acc_tr, prec_tr, recall_tr, f1_tr, c_k_tr, data_savepath)
    save_metrics('Val',acc_val, prec_val, recall_val, f1_val, c_k_val, data_savepath)
    save_metrics('Test',acc_te, prec_te, recall_te, f1_te, c_k_te, data_savepath)

    save_plots('Accuracy', acc_tr, acc_val, acc_te, RB, epochs, plot_savepath)
    save_plots('Precision', prec_tr, prec_val, prec_te, RB, epochs, plot_savepath)
    save_plots('Recall', recall_tr, recall_val, recall_te, RB, epochs, plot_savepath)
    save_plots('F1 score', f1_tr, f1_val, f1_te, RB, epochs, plot_savepath)
    save_plots('Cohen Kappa', c_k_tr, c_k_val, c_k_te, RB, epochs, plot_savepath)

def arrange_split_indices_without_wastage(l,bs):
    if l % bs == 0 or l % bs != 1:
        return torch.arange(0,l,bs), False
    else:
        t = torch.arange(0,l,bs)
        t = t[:-1]
        return t, True

def predict(lr, ep, batch_len, optim, model, x_train, x_val, x_test, y_train, y_val, y_test, experiment_number, run, set_number, fold):
    device = get_device()
    best_score = 0
    loss_function = nn.BCEWithLogitsLoss()
    learning_rate = lr  # 0.0005
    optimizer = optim(model.parameters(), lr=learning_rate)

    df_train = {"Loss": [], "Accuracy": []}
    df_val = {"Loss": [], "Accuracy": []}
    df_test = {"Loss": [], "Accuracy": []}
    model.to(device)
    epochs = ep  # 100
    batch_size = batch_len 
    batch_start_train, train_mis_flag = arrange_split_indices_without_wastage(len(x_train), batch_size)
    batch_start_val, val_mis_flag = arrange_split_indices_without_wastage(len(x_val), batch_size)
    batch_start_test, test_mis_flag = arrange_split_indices_without_wastage(len(x_test), batch_size)
    train_numbers = {}
    val_numbers = {}
    test_numbers = {}

    for epoch in range(epochs):
        
        correct_train_preds = 0
        train_loss_list = []
        epoch_train_actual = []
        epoch_train_predicted = []
        epoch_val_actual = []
        epoch_val_predicted = []
        epoch_test_actual = []
        epoch_test_predicted = []

        with tqdm.tqdm(
            batch_start_train, 
            unit="batch", 
            mininterval=0, 
            disable=True, 
            desc="Train"
        ) as bar:
            bar.set_description(f"Train Epoch {epoch}")
            for start in bar:
                if train_mis_flag and start == batch_start_train[-1]:
                    X_batch = x_train[start : start + batch_size*2 + 1].to(device)
                    y_batch = y_train[start : start + batch_size*2 + 1].to(device)
                else:
                    X_batch = x_train[start : start + batch_size].to(device)
                    y_batch = y_train[start : start + batch_size].to(device)
                train_pred = model(X_batch)  # forward pass
                #print(y_batch[:,-1,:].size(), train_pred.size())
                train_loss = loss_function(train_pred, y_batch[:,-1,:]) #last output of lstm, i.e, prediction for current frame/timestep
                train_loss_list.append(float(train_loss))
                train_pred = train_pred.clamp(0, 1)
                optimizer.zero_grad()
                train_loss.backward()  # backward pass
                optimizer.step()  # update weights
                correct_train_preds += (train_pred.round() == y_batch[:,-1,:]).float().sum().cpu()   
                epoch_train_actual.extend(y_batch[:,-1,:].cpu().detach().numpy())
                epoch_train_predicted.extend(train_pred.round().cpu().detach().numpy())

        train_numbers[f'{epoch}_actual'] = epoch_train_actual
        train_numbers[f'{epoch}_predicted'] = epoch_train_predicted

        df_train["Loss"].append(np.mean(train_loss_list))
        df_train["Accuracy"].append(100 * correct_train_preds / (len(x_train)))
        print(
            "Train Epoch [{}/{}], Accuracy: {:0.4f}; Loss: {:.4f}, lr: {}".format(
                epoch + 1,
                epochs,
                100 * correct_train_preds / len(x_train),
                np.mean(train_loss_list),
                optimizer.param_groups[0]["lr"]
            )
        )

        with torch.no_grad():
            correct_val_preds = 0
            val_loss_list = []
            with tqdm.tqdm(
                batch_start_val,
                unit="batch",
                mininterval=0,
                disable=True,
                desc="Val"
            ) as bar1:
                bar1.set_description(f"Val Epoch {epoch}")
                for start in bar1:
                    if val_mis_flag and start == batch_start_val[-1]:
                        X_batch = x_val[start : start + batch_size*2 + 1].to(device)
                        y_batch = y_val[start : start + batch_size*2 + 1].to(device)
                    else:
                        X_batch = x_val[start : start + batch_size].to(device)
                        y_batch = y_val[start : start + batch_size].to(device)
                    val_pred = model(X_batch)
                    val_loss = loss_function(val_pred, y_batch[:,-1,:])
                    val_loss_list.append(float(val_loss))
                    val_pred = val_pred.clamp(0, 1)
                    correct_val_preds += (val_pred.round() == y_batch[:,-1,:]).float().sum().cpu()
                    epoch_val_actual.extend(y_batch[:,-1,:].cpu().detach().numpy())
                    epoch_val_predicted.extend(val_pred.round().cpu().detach().numpy())

        val_numbers[f'{epoch}_actual'] = epoch_val_actual
        val_numbers[f'{epoch}_predicted'] = epoch_val_predicted
                    
        df_val["Loss"].append(np.mean(val_loss_list))
        df_val["Accuracy"].append(100 * correct_val_preds / len(x_val))
        print(
            "Val Epoch [{}/{}], Accuracy: {:0.4f}; Loss: {:.4f}".format(
                epoch + 1,
                epochs,
                100 * correct_val_preds / len(x_val),
                np.mean(val_loss_list),
            )
        )

        with torch.no_grad():
            correct_test_preds = 0
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
                    if test_mis_flag and start == batch_start_test[-1]:
                        X_batch = x_test[start : start + batch_size*2 + 1].to(device)
                        y_batch = y_test[start : start + batch_size*2 + 1].to(device)
                    else:
                        X_batch = x_test[start : start + batch_size].to(device)
                        y_batch = y_test[start : start + batch_size].to(device)
                    test_pred = model(X_batch)
                    test_loss = loss_function(test_pred, y_batch[:,-1,:])
                    test_loss_list.append(float(test_loss))
                    test_pred = test_pred.clamp(0, 1)
                    correct_test_preds += (test_pred.round() == y_batch[:,-1,:]).float().sum().cpu()
                    # print('expected', y_batch[:,-1,:])
                    # print('predicted',test_pred.round())
                    epoch_test_actual.extend(y_batch[:,-1,:].cpu().detach().numpy())
                    epoch_test_predicted.extend(test_pred.round().cpu().detach().numpy())
            
            test_numbers[f'{epoch}_actual'] = epoch_test_actual
            test_numbers[f'{epoch}_predicted'] = epoch_test_predicted
            
            df_test["Loss"].append(np.mean(test_loss_list))
            df_test["Accuracy"].append(100 * correct_test_preds / len(x_test))
            print(
                "Test Epoch [{}/{}], Accuracy: {:0.4f}; Loss: {:.4f}".format(
                    epoch + 1,
                    epochs,
                    100 * correct_test_preds / len(x_test),
                    np.mean(test_loss_list),
                )
            )
        
        if (100 * correct_test_preds / (len(x_test))) > best_score:
            best_score = 100 * correct_test_preds / (len(x_test))
            print('New best test metric observed! Saving Model..')

            model_savepath = DIR_PATH / f'inference/lstm/experiment-{experiment_number}/run-{run}/set-{set_number}/fold-{fold}/models'
            if not os.path.exists(model_savepath):
                os.makedirs(model_savepath)
            
            lstm_utils.save_model(model,epoch,optimizer,model_savepath)
            
        #scheduler.step() #update learning rate ~ Per epoch

    return df_train, df_val, df_test, train_numbers, val_numbers, test_numbers

def main():

    LSTM_classifier = lstm_utils.LSTM_Classifier

    SEED = 7

    feat_set = ['AU06_r','AU12_r']

    extracted_data = lstm_utils.data_prep_new_pipeline(WINDOW_LENGTH, feat_set)

    _, _, _, val_pair_ids = generate_train_val_split(extracted_data, seed=SEED, sklearn=True, k=K)
    
    num_of_features = len(feat_set)

    experiment_number = 14

    experiment_savepath = DIR_PATH / f'inference/lstm/experiment-{experiment_number}'

    if not os.path.exists(experiment_savepath):
        os.makedirs(experiment_savepath)    

    logging.basicConfig(filename= experiment_savepath / 'info_log.txt', filemode='a', format='%(message)s',datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

    dt = datetime.datetime.now()

    logging.info(dt.strftime("%c"))

    logging.info(f"\n\n{K}-Fold Cross validation! \n")
    # Hyperparams = {
    #     "Learning rate": [0.01, 0.001, 0.0001, 0.00001], # , 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.000001
    #     "Epochs": [100, 250, 500, 1000, 1500], #, 200, 400, 500
    #     "Batch size": [8, 16, 32], # 8, 16, 32, 64 , 32, 64
    #     "Optimizers":[
    #                    {'Name': 'SGD', 'fn': optim.SGD},
    #                    {'Name': 'RMSprop', 'fn': optim.RMSprop},
    #                    {'Name': 'Adam', 'fn': optim.Adam},
    #                    {'Name': 'Adamax', 'fn': optim.Adamax},
    #                    {'Name': 'RAdam', 'fn': optim.RAdam}, 
    #     ]
    # }

    Hyperparams = {
        "Learning rate": [0.001, 0.0001], # , 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.000001
        "Epochs": [1000], #, 200, 400, 500
        "Batch size": [16, 32], # 8, 16, 32, 64 , 32, 64
        "Optimizers":[
                    {'Name': 'Adam', 'fn': optim.Adam},
                    {'Name': 'RMSprop', 'fn': optim.RMSprop}
        ]
    }

    run = 0
    for opti in Hyperparams["Optimizers"]:
        for lr in Hyperparams["Learning rate"]:
            for ep in Hyperparams["Epochs"]:
                for bs in Hyperparams["Batch size"]:

                    optim_name = opti['Name']

                    Model = LSTM_classifier(num_of_features)
                    
                    logging.info(f"Run - {run}! \n")
          
                    logging.info(f"Hyperparams: Optimizer- {opti}, Learning rate- {lr}, Epochs- {ep}, Batch size- {bs} \n\n")

                    logging.info(f"\n\n{K}-Fold Cross validation! \n")
                    
                    for i,fold_test_pair_id in enumerate(val_pair_ids):

                        set = i+1
                        
                        logging.info(f"\n Fold - {set} \n")
                        
                        fold_val_pair_ids = val_pair_ids.copy()
                        
                        fold_val_pair_ids.remove(fold_test_pair_id)

                        logging.info(f"\nTest pair IDs: {fold_test_pair_id} \n")
                        
                        fold_train_pair_ids  = get_clean_fold_train_pair_ids(extracted_data.pair_id.unique(), fold_test_pair_id, fold_val_pair_ids) 

                        fold = 1

                        logging.info(f"\n\n{K-1}-Fold cross-validation with rest of the set! \n")
                        
                        for train_p_ids, val_p_ids in zip(fold_train_pair_ids, fold_val_pair_ids):
                            
                            logging.info(f"\n\nSub-fold: {fold}! \n")
                            
                            logging.info(f"\nTrain pair IDs: {train_p_ids} \n")

                            logging.info(f"\nVal pair IDs: {val_p_ids} \n")

                            X_train, X_val, X_test, Y_train, Y_val, Y_test, split_stats = lstm_utils.process_foldwise_data_with_custom_labels( WINDOW_LENGTH, feat_set, train_p_ids, val_p_ids, fold_test_pair_id, overlapping_windows=False )

                            RB = {
                                'Train': split_stats['Train RB'],
                                'Val': split_stats['Val RB'],
                                'Test': split_stats['Test RB']
                            }

                            logging.info(f"\n\nRandom Baselines: Train - {split_stats['Train RB']}, Val - {split_stats['Val RB']}, Test - {split_stats['Test RB']}, \n")
                            
                            print(X_train.shape, X_val.shape, X_test.shape)

                            dataframe_train, dataframe_val, dataframe_test, train_numbers, val_numbers, test_numbers= predict(
                                lr, ep, bs, opti['fn'], Model, X_train, X_val, X_test, Y_train, Y_val, Y_test, experiment_number, run, set, fold
                            )
                            
                            save_path = DIR_PATH / f'inference/lstm/experiment-{experiment_number}/run-{run}/set-{set}/fold-{fold}'
                            
                            if not os.path.exists(save_path / 'data'):
                                os.makedirs(save_path / 'data')
                            
                            df_save_path = save_path / f'data/Run - {run}_{lr}_{ep}_{bs}_{optim_name}.csv'

                            results_df=pd.DataFrame()
                            results_df['Train acc'] = dataframe_train["Accuracy"]
                            results_df['Val acc'] = dataframe_val['Accuracy']
                            results_df['Test acc'] = dataframe_test["Accuracy"]
                            results_df['Train loss'] = dataframe_train["Loss"]
                            results_df['Val loss'] = dataframe_val["Loss"]
                            results_df['Test loss'] = dataframe_test["Loss"]
                            results_df.to_csv(df_save_path, index=False)

                            if not os.path.exists(save_path / 'plots'):
                                os.makedirs(save_path / 'plots')
                            
                            extract_and_save__metrics_and_plots(train_numbers, val_numbers, test_numbers, ep, RB, save_path)

                            print("Run:{0} ; Train acc: {1} ; Test acc: {2} for ( {3},{4},{5} )".format(
                                run,
                                dataframe_train["Accuracy"][-1],
                                dataframe_test["Accuracy"][-1],
                                lr,
                                ep,
                                bs
                            ))

                            fold += 1

                    run += 1

if __name__ == '__main__':
    main()