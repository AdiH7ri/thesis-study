import os
from pathlib import Path
import pandas as pd
import more_itertools

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, f1_score, recall_score, precision_score, matthews_corrcoef

import opensmile
import librosa

import random
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, StandardScaler
import torch

DIR_PATH = Path(__file__).parent
AUD_FILE_PATH = Path.joinpath(DIR_PATH.parent, 'headset-audio')
PROCESSED_AUD_FILE_PATH = Path.joinpath(DIR_PATH, 'processed-data') 
FPS = 25
CROPPED_SECONDS_FROM_BEGINNING = 3.48

FEATURE_SETS = {
    "GeM": {
            "extractor": opensmile.Smile(
                feature_set=opensmile.FeatureSet.GeMAPSv01b,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                multiprocessing=True,
            ),
            "features": lambda x: x.feature_names,
            },
    "eGe": {
            "extractor": opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                multiprocessing=True,
            ),
            "features": lambda x: x.feature_names,
            },
    "emob": {
            "extractor": opensmile.Smile(
                feature_set=opensmile.FeatureSet.emobase,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                multiprocessing=True,
            ),
            "features": lambda x: x.feature_names,
            },
    "GeM_5": {
            "extractor": opensmile.Smile(
                feature_set=opensmile.FeatureSet.GeMAPSv01b,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                multiprocessing=True,
            ),
            "features": lambda x: x.feature_names[:5],
            },
    "GeM_13": {
            "extractor": opensmile.Smile(
                feature_set=opensmile.FeatureSet.GeMAPSv01b,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                multiprocessing=True,
            ),
            "features": lambda x: x.feature_names[-13:],
            },
    "eGe_10": {
            "extractor": opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                multiprocessing=True,
            ),
            "features": lambda x: x.feature_names[:10],
            },
    "eGe_13": {
            "extractor": opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                multiprocessing=True,
            ),
            "features": lambda x: x.feature_names[-13:],
            },
}

MODELS = {
    "Logistic_Regression": {
        "function": LogisticRegression(),
        "param_grid": [
            {
                "Logistic_Regression__solver": ["saga"],
                "Logistic_Regression__penalty": ["l1", "l2"],
                "Logistic_Regression__tol": [1e-2, 1e-3, 1e-4, 1e-5],
            }
        ],
    },
    "SVM": {
        "function": SVC(),
        "param_grid": [
            {"SVM__kernel": ["poly"], "SVM__degree": [2, 3, 4, 5, 6, 7, 8]}, #, 6, 7, 8, 9, 10, 15
            {"SVM__kernel": ["linear", "rbf", "sigmoid"]},
        ],
    },
    "RF": {
        "function": RandomForestClassifier(),
        "param_grid": [
            {
                "RF__n_estimators": [2, 5, 10, 20],
                "RF__criterion": ["gini", "entropy", "log_loss"],
                "RF__max_depth": [i for i in range (2,16)], #, 15, 20, 25
                "RF__max_features": [0.1, 0.2, "sqrt", "log2"],
            }
        ],
    },
    "KNN": {
        "function": KNeighborsClassifier(),
        "param_grid": [{"KNN__n_neighbors": [1, 2, 3, 4, 5, 7, 9, 11]}], # , 9, 11, 13, 15
    },
}
# Solver saga supports only 'l2' or 'none' penalties
# Solver newton-cholesky supports only 'l2' or 'none' penalties


def extract_raw_data(extractor, feature_set_name, features, summary_stat = None, custom_window_length_factor = None, timestamp = False):

# --------------------------- Reading the processed audio file names from disk ------------------------ #    
    audio_files = PROCESSED_AUD_FILE_PATH.glob("*.mp3")
    audio_file_names = [i.stem for i in audio_files]
    audio_file_names.sort()
    
    child_link_csv = pd.read_csv('/media/chagan/2023-ELECTRA-field-MarcSchoolStudy/headset-audio/vidfile_link.csv')


    if summary_stat is None:
        
        if custom_window_length_factor is None:
            extracted_data_savepath = DIR_PATH / f"extracted-data/raw-extract/{feature_set_name}-extract.csv"
        else:
            extracted_data_savepath = DIR_PATH / f"extracted-data/raw-extract/{feature_set_name}-cwlf({custom_window_length_factor})-extract.csv"
        
        agg_aud_df = pd.DataFrame()
        if extracted_data_savepath.is_file():
            print('Data already exists in disk, reading..\n')
            if timestamp:
                agg_aud_df = pd.read_csv(extracted_data_savepath, usecols=features+['pair_id','round','child_id','condition', 'timestamp'])    
                return agg_aud_df
            else:
                agg_aud_df = pd.read_csv(extracted_data_savepath, usecols=features+['pair_id','round','child_id','condition'])    
                return agg_aud_df         

        
        else:
            all_features = FEATURE_SETS[feature_set_name]['features'](extractor)
            print('Data does not exist in disk, reading audio files and extracting to a csv..\n')
            for audio_file in audio_file_names:
                # ['headset', 'audio', 'N249', 'planning', '1', 'child', '1', 'processed']
                splitted_name = audio_file.split("-")
                pair_id = splitted_name[2]
                round = int(splitted_name[4])
                child = int(splitted_name[6])
                child_id = child_link_csv['child_id'].loc[ (child_link_csv['pair_id']==pair_id) & (child_link_csv['round']==round) & (child_link_csv['child']==child) ].reset_index(drop=True)[0] #get child_id from 'vidfile_link.csv'

                x, sr = librosa.load(
                    Path.joinpath(PROCESSED_AUD_FILE_PATH, audio_file + ".mp3"), sr=None
                )

                if custom_window_length_factor == None:
                    aud_df = pd.DataFrame(extractor.process_signal(x, sr))
                else:
                    aud_df = pd.DataFrame(extractor.process_signal(x, sr * custom_window_length_factor))

                feature_outputs = aud_df[(col for col in all_features)].astype(float).reset_index(drop=True)
                feature_outputs['timestamp'] = [x*(1 / FPS) + CROPPED_SECONDS_FROM_BEGINNING for x in range(0,aud_df.shape[0])]
                feature_outputs['pair_id']= pair_id
                feature_outputs['round']= round
                feature_outputs['child_id']= child_id
                feature_outputs["condition"] = "positive" if (pair_id[0] == "P") else "negative"

                agg_aud_df = pd.concat([agg_aud_df, feature_outputs]).reset_index(drop=True)

            agg_aud_df.to_csv(extracted_data_savepath, index=False)
            
            print(f'Extracted feature csv {extracted_data_savepath} saved successfully!\n')
            if timestamp:
                return agg_aud_df[features+['pair_id','round','child_id','condition', 'timestamp']]
            else:
                return agg_aud_df[features+['pair_id','round','child_id','condition']]
    else:
        if not os.path.exists(DIR_PATH / f"extracted-data/summary-statistics-{summary_stat}"):
            os.makedirs(DIR_PATH / f"extracted-data/summary-statistics-{summary_stat}")
        
        if custom_window_length_factor is None:
            extracted_data_savepath = DIR_PATH / f"extracted-data/summary-statistics-{summary_stat}/{feature_set_name}-extract.csv"
        else:
            extracted_data_savepath = DIR_PATH / f"extracted-data/summary-statistics-{summary_stat}/{feature_set_name}-cwlf({custom_window_length_factor})-extract.csv"
        
        if extracted_data_savepath.is_file():
            print('Data already exists in disk, reading..\n')
            agg_aud_df = pd.read_csv(extracted_data_savepath)    
            return agg_aud_df
        
        else:
            print('Data does not exist in disk, reading audio files and extracting to a csv..\n')
            all_features = FEATURE_SETS[feature_set_name]['features'](extractor)
            print(all_features)
            agg_aud_df = pd.DataFrame(columns=['pair_id','round', 'child_id', 'condition'] + all_features)

            for audio_file in audio_file_names:
                # ['headset', 'audio', 'N249', 'planning', '1', 'child', '1', 'processed']
                splitted_name = audio_file.split("-")
                pair_id = splitted_name[2]
                round = int(splitted_name[4])
                child = int(splitted_name[6])
                child_id = child_link_csv['child_id'].loc[ (child_link_csv['pair_id']==pair_id) & (child_link_csv['round']==round) & (child_link_csv['child']==child) ].reset_index(drop=True)[0] #get child_id from 'vidfile_link.csv'
                condition = "positive" if (pair_id[0] == "P") else "negative"

                x, sr = librosa.load(
                    Path.joinpath(PROCESSED_AUD_FILE_PATH, audio_file + ".mp3"), sr=None
                )
                
                if custom_window_length_factor == None:
                    aud_df = pd.DataFrame(extractor.process_signal(x, sr))
                else:
                    aud_df = pd.DataFrame(extractor.process_signal(x, sr * custom_window_length_factor))

                agg_aud_df.loc[len(agg_aud_df)] = [splitted_name[2], splitted_name[4], child_id, condition ] + aud_df[ all_features ].describe().loc[summary_stat].to_list()   #.mean().to_list()


            agg_aud_df.to_csv(extracted_data_savepath, index=False)
            
            print(f'Extracted feature csv {extracted_data_savepath} saved successfully!\n')

            return agg_aud_df[features+['pair_id','round','child_id','condition']]            

def process_data(split_ratio, data, features):

    pair_id_list = list(data["pair_id"].unique())  # get unique pair_ids

    flag=True

    tot_size=data.shape[0]

    #Rejection sampling
    split_stats={}

    while flag:

        random.shuffle(pair_id_list)
        split_index = round(split_ratio * len(pair_id_list))
        train_pair_ids = pair_id_list[: split_index]
        test_pair_ids = pair_id_list[split_index:]
        
        train_length = data.loc[
        data["pair_id"].isin(pair_id_list[: split_index]) #split ratio
        ].reset_index(drop=True).shape[0]

        df_sliced_with_train_pid = data.loc[data["pair_id"].isin(train_pair_ids)].reset_index(drop=True)
        df_sliced_with_test_pid = data.loc[data["pair_id"].isin(test_pair_ids)].reset_index(drop=True)
        train_size = df_sliced_with_train_pid.shape[0]

        test_p_size = len([p for p in df_sliced_with_test_pid['pair_id'] if p[0] == 'P'])
        test_n_size = len([n for n in df_sliced_with_test_pid['pair_id'] if n[0] == 'N'])        
        
        condition_1 = train_size/tot_size > (split_ratio-0.02)
        condition_2 = train_size/tot_size < (split_ratio+0.02)
        condition_3 = len([d for d in pair_id_list[ : split_index] if d[0]=='N']) - len([d for d in pair_id_list[: split_index] if d[0]=='P']) < 2
        condition_4 = 0.5 <= test_p_size / df_sliced_with_test_pid.shape[0] and test_p_size / df_sliced_with_test_pid.shape[0] <= 0.52
        
        if condition_1 and  condition_2 and condition_3 and condition_4:
            flag=False
    
    split_stats['Split ratio']=train_length/tot_size
    split_stats['IDs in Train']=pair_id_list[: split_index]
    split_stats['IDs in Test']=pair_id_list[split_index : ]
    split_stats['Number of Ns in train']=len([d for d in pair_id_list[: split_index] if d[0]=='N'])
    split_stats['Number of Ps in train']=len([d for d in pair_id_list[: split_index] if d[0]=='P'])
    split_stats['Number of Ns in test']=len([d for d in pair_id_list[split_index :] if d[0]=='N'])
    split_stats['Number of Ps in test']=len([d for d in pair_id_list[split_index :] if d[0]=='P'])

    scaler=StandardScaler()
    
    x_train = data.loc[
        data["pair_id"].isin(pair_id_list[: split_index])
    ].reset_index(drop=True)
    
    scaler.fit(x_train[features])
    x_train = scaler.transform(x_train[features])
    y_train = data.loc[
        data["pair_id"].isin(pair_id_list[: split_index])
    ].reset_index(drop=True)
    y_train = y_train["pair_id"].apply(lambda x: 1.0 if (x[0] == "P") else 0.0)

    x_test = data.loc[
        data["pair_id"].isin(pair_id_list[split_index :])
    ].reset_index(drop=True)
    x_test = scaler.transform(x_test[features])

    y_test = data.loc[
        data["pair_id"].isin(pair_id_list[split_index :])
    ].reset_index(drop=True)
    y_test = y_test["pair_id"].apply(lambda x: 1.0 if (x[0] == "P") else 0.0)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    return x_train, x_test, y_train, y_test, split_stats

def generate_train_val_split(df, seed, sklearn = False, k=8,  level=1, outliers= None):
    train_idx, val_idx, train_pair_ids, val_pair_ids = list(), list(), list(), list()
    
    random.seed(seed)

    if outliers is None:
        pair_id_list = list(set(df["pair_id"]))
        pair_id_list.sort()
        random.shuffle(pair_id_list)
    else:
        pair_id_list = [pair_id for pair_id in list(set(df["pair_id"])) if pair_id not in outliers]
        pair_id_list.sort()
        random.shuffle(pair_id_list)
    if not sklearn:
        p_y2 = [x for x in pair_id_list if x[0]=='P' and x[1]=='2']
        n_y2 = [x for x in pair_id_list if x[0]=='N' and x[1]=='2']
        p_y3 = [x for x in pair_id_list if x[0]=='P' and x[1]=='3']
        n_y3 = [x for x in pair_id_list if x[0]=='N' and x[1]=='3']
        
        if level == 1:
            p = p_y2 + p_y3
            n = n_y2 + n_y3
            print(len(p),len(n))
            for i, j in zip (more_itertools.batched(p, int(len(p)/k)), more_itertools.batched(n, int(len(n)/k))):
                val_pair_id = list(i+j)
                val_pair_ids.append(val_pair_id)
                val_idx.append(df.loc[ df["pair_id"].isin(val_pair_id)].index.to_list())

                train_pair_id = [x for x in pair_id_list if x not in val_pair_id]
                train_pair_ids.append(train_pair_id)
                train_idx.append(df.loc[ df["pair_id"].isin(train_pair_id)].index.to_list())
            
            return train_idx, val_idx, train_pair_ids, val_pair_ids
        else:

            for i, j, k, l in zip (more_itertools.batched(p_y2, int(len(p_y2)/k)), more_itertools.batched(p_y3, int(len(p_y3)/k)), more_itertools.batched(n_y2, int(len(n_y2)/k)), more_itertools.batched(n_y3, int(len(n_y3)/k))):
                val_pair_id = list(i+j+k+l)
                val_pair_ids.append(val_pair_id)
                val_idx.append(df.loc[ df["pair_id"].isin(val_pair_id)].index.to_list())

                train_pair_id = [x for x in pair_id_list if x not in val_pair_id]
                train_pair_ids.append(train_pair_id)
                train_idx.append(df.loc[ df["pair_id"].isin(train_pair_id)].index.to_list())

            return train_idx, val_idx, train_pair_ids, val_pair_ids        
    else:

        X = df[[x for x in df.columns if x not in ['pair_id','round','child_id','condition']]]
        y = df['condition']
        groups = df.pair_id

        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)

        for train, val in cv.split(X, y, groups):
            train_idx.append(train)
            train_pair_ids.append( list(set(df['pair_id'].loc[train])) )
            
            val_idx.append(val.tolist())
            val_pair_ids.append( list(set(df['pair_id'].loc[val])) )
        
        return train_idx, val_idx, train_pair_ids, val_pair_ids 

def generate_clean_train_and_val_idx(df, pid_list, test, v_pids):
    t_pid = []
    t_idx = []

    v_idx = []

    for i in v_pids:
        clean = [x for x in pid_list if x not in (test + i)]
        t_pid.append(clean)
        t_idx.append( df.loc[ df["pair_id"].isin(clean)].index.to_list() )
        v_idx.append( df.loc[ df["pair_id"].isin(i)].index.to_list() )
    return t_idx, v_idx, t_pid

def return_indices_as_list(train_indices, val_indices):
    tr_lst = []
    val_lst = []

    for (x,y) in zip(train_indices, val_indices):
        tr_lst.extend(x)
        val_lst.extend(y)
    
    return tr_lst, val_lst

def generate_train_val_test_idx(df, seed, k=8, test_split_ratio=0.2, level=1, outliers= None):

    train_idx, val_idx, test_idx, train_pair_ids, val_pair_ids, test_pair_ids = list(), list(), list(), list(), list(), list()

    if outliers is None:
        pair_id_list = list(set(df["pair_id"]))
    else:
        pair_id_list = [pair_id for pair_id in list(set(df["pair_id"])) if pair_id not in outliers]

    #Rejection sampling

    flag = True
    df_size = df.shape[0]

    while flag:
        
        random.shuffle(pair_id_list)
        split_index = round(test_split_ratio * len(pair_id_list))

        test_pair_ids = pair_id_list[: split_index]
        train_pair_ids = pair_id_list[split_index:]

        df_sliced_with_train_pid = df.loc[df["pair_id"].isin(train_pair_ids)].reset_index(drop=True)
        df_sliced_with_test_pid = df.loc[df["pair_id"].isin(test_pair_ids)].reset_index(drop=True)
        train_size = df_sliced_with_train_pid.shape[0]

        test_p_size = len([p for p in df_sliced_with_test_pid['pair_id'] if p[0] == 'P'])
        test_n_size = len([n for n in df_sliced_with_test_pid['pair_id'] if n[0] == 'N'])        
        
        condition_1 = train_size/df_size > (test_split_ratio-0.02)
        condition_2 = train_size/df_size < (test_split_ratio+0.02)
        condition_3 = len([d for d in pair_id_list[ : split_index] if d[0]=='N']) - len([d for d in pair_id_list[: split_index] if d[0]=='P']) < 2
        condition_4 = 0.5 <= test_p_size / df_sliced_with_test_pid.shape[0] and test_p_size / df_sliced_with_test_pid.shape[0] <= 0.52
        
        if condition_1 and  condition_2 and condition_3 and condition_4:
            flag=False

    test_idx = df.loc[ df["pair_id"].isin(test_pair_ids)].index.to_list()

    df_sliced = df.loc[df["pair_id"].isin(train_pair_ids)].reset_index(drop=True)

    random.seed(seed)
    
    pair_id_list = [x for x in pair_id_list if x not in test_pair_ids]

    pair_id_list.sort()

    train_pair_ids.clear()

    random.shuffle(pair_id_list)

    p_y2 = [x for x in pair_id_list if x[0]=='P' and x[1]=='2']
    n_y2 = [x for x in pair_id_list if x[0]=='N' and x[1]=='2']
    p_y3 = [x for x in pair_id_list if x[0]=='P' and x[1]=='3']
    n_y3 = [x for x in pair_id_list if x[0]=='N' and x[1]=='3']
    
    if level == 1:
        p = p_y2 + p_y3
        n = n_y2 + n_y3

        for i, j in zip (more_itertools.batched(p, int(len(p)/k)), more_itertools.batched(n, int(len(n)/k))):
            val_pair_id = list(i+j)
            val_pair_ids.append(val_pair_id)
            val_idx.append(df_sliced.loc[ df_sliced["pair_id"].isin(val_pair_id)].index.to_list())

            train_pair_id = [x for x in pair_id_list if x not in val_pair_id]
            train_pair_ids.append(train_pair_id)
            train_idx.append(df_sliced.loc[ df_sliced["pair_id"].isin(train_pair_id)].index.to_list())
        
        return train_idx, val_idx, test_idx, train_pair_ids, val_pair_ids, test_pair_ids
    
    else:

        for i, j, k, l in zip (more_itertools.batched(p_y2, int(len(p_y2)/k)), more_itertools.batched(p_y3, int(len(p_y3)/k)), more_itertools.batched(n_y2, int(len(n_y2)/k)), more_itertools.batched(n_y3, int(len(n_y3)/k))):
            val_pair_id = list(i+j+k+l)
            val_pair_ids.append(val_pair_id)
            val_idx.append(df_sliced.loc[ df_sliced["pair_id"].isin(val_pair_id)].index.to_list())

            train_pair_id = [x for x in pair_id_list if x not in val_pair_id]
            train_pair_ids.append(train_pair_id)
            train_idx.append(df_sliced.loc[ df_sliced["pair_id"].isin(train_pair_id)].index.to_list())

        return train_idx, val_idx, test_idx, train_pair_ids, val_pair_ids, test_pair_ids                

def calculate_random_baseline_from_splits(df, train_pids, val_pids, test_pids):
    
    tot_tr = 0
    tot_te = 0
    for x,y in zip(train_pids,val_pids):

        train_idx_p = len([p for p in df['pair_id'].loc[ df['pair_id'].isin(x) ] if p[0] == 'P'])
        train_idx_n = len([n for n in df['pair_id'].loc[ df['pair_id'].isin(x) ] if n[0] == 'N'])
        val_idx_p = len([p for p in df['pair_id'].loc[ df['pair_id'].isin(y) ] if p[0] == 'P'])
        val_idx_n = len([n for n in df['pair_id'].loc[ df['pair_id'].isin(y) ] if n[0] == 'N'])

        tot_tr += train_idx_p/(train_idx_p+train_idx_n)
        tot_te += val_idx_p/(val_idx_p+val_idx_n)

    test_idx_p = len([p for p in df['pair_id'].loc[ df['pair_id'].isin(test_pids) ] if p[0] == 'P'])
    test_idx_n = len([n for n in df['pair_id'].loc[ df['pair_id'].isin(test_pids) ] if n[0] == 'N'])

    train_rb = tot_tr/len(train_pids)
    val_rb = tot_te/len(val_pids)
    test_rb = test_idx_p/(test_idx_p + test_idx_n)

    return train_rb, val_rb, test_rb

def get_metrics(actual, predicted):

    return accuracy_score(actual, predicted), precision_score(actual, predicted), recall_score(actual, predicted), f1_score(actual, predicted), cohen_kappa_score(actual, predicted), matthews_corrcoef(actual, predicted)


def get_clean_foldwise_df(df, modality='Uni'):
       metrics = ['CV_Val_accuracy', 'CV_Val_RB','Test_accuracy_score', 'Test_precision_score', 'Test_recall_score',
              'Test_F1_score', 'Test_Cohen_Kappa_score', 'Test_MCC_score', 'Test_RB']

       if modality == 'Uni':
              
              new_df = pd.DataFrame(columns=['K_fold', 'Model'] + [ 'Mean_'+ x for x in metrics ] + [ 'Std_'+ x for x in metrics ] + ['Features'])

              for K in df.K_fold.unique():
              
                     k_df = df.loc[ df['K_fold'] == K ]

                     for feat in k_df.Features.unique():
                            
                            fold_results_df_combined = k_df.loc[ (k_df['Features'] == feat) ]

                            for model in fold_results_df_combined.Model.unique():
                            
                                   fold_result_df_model = fold_results_df_combined.loc[ fold_results_df_combined['Model'] == model ]

                                   #print(fold_result_df_model.columns) #[metrics].mean()) #.loc['Test_accuracy_score']
                                   #print([K, model] + [ fold_result_df_model[metrics].mean().loc[x] for x in metrics ] + [ fold_result_df_model[metrics].std().loc[x] for x in metrics ] + [feat])

                                   new_df.loc[len(new_df)] = [K, model] + [ fold_result_df_model[metrics].mean().loc[x] for x in metrics ]  + [ fold_result_df_model[metrics].std().loc[x] for x in metrics ] + [feat]

              new_df.sort_values(by=['Mean_Test_MCC_score'], ascending=False, inplace=True)

              return new_df
       else:
              new_df = pd.DataFrame(columns=['K_fold', 'Model'] + [ 'Mean_'+ x for x in metrics ] + [ 'Std_'+ x for x in metrics ]  + ['Aud_features', 'Vid_features'])
              
              
              for aud_f in df.Aud_Features.unique():

                for vid_f in df.Vid_Features.unique():

                    feat_df_combined = df.loc[ (df['Aud_Features']  == aud_f ) & (df['Vid_Features'] == vid_f) ]
                    
                    for K in feat_df_combined.K_fold.unique():
                        
                        k_df = feat_df_combined.loc[ feat_df_combined['K_fold'] == K ]

                        for model in k_df.Model.unique():
                        
                            fold_result_df_model = k_df.loc[ k_df['Model'] == model ]

                            #print(fold_result_df_model.columns) #[metrics].mean()) #.loc['Test_accuracy_score']
                            #print([K, model] + [ fold_result_df_model[metrics].mean().loc[x] for x in metrics ] + [aud_f, vid_f])

                            new_df.loc[len(new_df)] = [K, model] + [ fold_result_df_model[metrics].mean().loc[x] for x in metrics ] + [ fold_result_df_model[metrics].std().loc[x] for x in metrics ]  + [aud_f, vid_f]
              
              new_df.sort_values(by=['Mean_Test_MCC_score'], ascending=False, inplace=True)           

              return new_df
       

# def main():
#     feature_set_name = 'emob' #'GeM', 'eGe', 

#     extractor = FEATURE_SETS[feature_set_name]['extractor']
#     #features = feature_set_items["features"](extractor)

#     features = [ 'pcm_loudness_sma',
#                 'pcm_intensity_sma',
#                 'F0env_sma'
#                 ] #Custom features

#     aud_mean_data = extract_raw_data(extractor, feature_set_name, features)

#     train_test_split_ratio = 0.6
#     X_train, X_test, Y_train, Y_test, stats = process_data(train_test_split_ratio, aud_mean_data, features)

# if __name__ == "__main__":
#     main()