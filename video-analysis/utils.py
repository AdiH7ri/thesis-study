import random
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, StandardScaler
import torch
import numpy as np
import pandas as pd
import re
from pathlib import Path


#----------------------------------Global Variables for defining fixed paths -----------------------------------#

DIR_PATH = Path(__file__).parent

DATA_SAVEPATH = DIR_PATH / 'data'

VIDEO_DIRS_OLD = [
    Path("/media/chagan/2023-ELECTRA-field-MarcSchoolStudy/scratch/backup/left-cam")  # , Path("./media/chagan/2023-ELECTRA-field-MarcSchoolStudy/scratch/backup/right-cam"), Path("/media/chagan/2023-ELECTRA-field-MarcSchoolStudy/scratch/backup/frontal-cam")
] 

VIDEO_DIRS_NEW = {
    'left': Path("/media/chagan/2023-ELECTRA-field-MarcSchoolStudy/data-backup/20231127T213846/left-cam"),  # , Path("./media/chagan/2023-ELECTRA-field-MarcSchoolStudy/scratch/backup/right-cam"), Path("/media/chagan/2023-ELECTRA-field-MarcSchoolStudy/scratch/backup/frontal-cam")
    'right': Path("/media/chagan/2023-ELECTRA-field-MarcSchoolStudy/data-backup/20231127T213846/right-cam")
}



def generate_csv_list(key: str, position = 'left') -> list[Path]:
    '''

    Args: 'old' or 'new' depending on which set of csv's we would like to work with.

    Returns: List[csv_name(s)]
    
    '''
    csv_list = []
    if key=='old':
        for video_dir in VIDEO_DIRS_OLD:
            assert video_dir.is_dir(), f"Video dir not found: {video_dir}"
            csvs = [
                file
                for file in video_dir.iterdir()
                if (file.suffix.lower() == ".csv")
                and ("dummy" not in file.stem)
                and (file.stem.split("-")[5] == "face")
            ]
            csvs.sort()
            csv_list.extend(csvs)

        return csv_list

    else:
        video_dir = VIDEO_DIRS_NEW[position]
        assert video_dir.is_dir(), f"Video dir not found: {video_dir}"
        csvs = [
            file
            for file in video_dir.iterdir()
            if (file.suffix.lower() == ".csv")
            and ("dummy" not in file.stem)
            and (file.stem.split("-")[5] == "consolidated")
            and (file.stem.split("-")[6] == "face")
        ]
        csvs.sort()
        csv_list.extend(csvs)

        return csv_list



def extract_from_raw_data(features: list[str], summary_stat: str= 'None', timestamp: bool = False, position: str= 'left') -> pd.DataFrame:
    '''
    Reads data from the disk and returns a Dataframe containing feature columns present in 'features'. 
    Also saves the aggregated csv to the disk it it doesn't exist already (Aggregated csv contains all the features returned by Openface)

    ## Parameters:
        - features: list() containing names of features (str)
    
        - summary_stat: 'mean', '75p', '50p', '25p' or 'None'
            - Different summary statistics techniques available in 'pd.Dataframe.describe()'
            - 'None' is used when the raw frame wise extract, per round, is needed.
    ## Returns:
        Dataframe containing the features mentioned in the parameter 'features'
    '''
    csv_list = list()

    video_dir = VIDEO_DIRS_NEW[position]
    assert video_dir.is_dir(), f"Video dir not found: {video_dir}"
    csvs = [
        file
        for file in video_dir.iterdir()
        if (file.suffix.lower() == ".csv")
        and ("dummy" not in file.stem)
        and (file.stem.split("-")[5] == "consolidated")
        and (file.stem.split("-")[6] == "face")
    ]
    csvs.sort()
    csv_list.extend(csvs)

    if summary_stat == 'None':
        
        extracted_data_savepath = DATA_SAVEPATH / f"raw-extract-{position}.csv"
        df_concat = pd.DataFrame()

        if extracted_data_savepath.is_file():
            print('Data already exists in disk, reading..\n')
            if timestamp:
                df_concat = pd.read_csv(extracted_data_savepath, usecols=features+['pair_id','round','child_id','condition', 'timestamp'])    
                return df_concat
            else:
                df_concat = pd.read_csv(extracted_data_savepath, usecols=features+['pair_id','round','child_id','condition'])   
                return df_concat   
        
        else:
            print('Data does not exist in disk, creating a new extract csv..\n')
            for csv in csv_list:
                data = pd.read_csv(csv)
                data = data.loc[data["success"] == 1]  # filtering
                child_ids =  data['child_id'].unique()
                cols = [x for x in data.columns if x not in ['frame','child_id','confidence','success']]
                for child in child_ids:
                    dat = data.loc[data['child_id'] == child].reset_index(drop = True)
                    au_r_values = (
                        dat[(col for col in cols)] #dat.columns if re.match("AU\d{2}_r", col)
                        .astype(float)
                        .reset_index(drop=True)
                    )
                    au_r_values["pair_id"] = pair_id = csv.stem.split("-")[2]
                    au_r_values["round"] = csv.stem.split("-")[4]
                    au_r_values["condition"] = (
                        "positive" if (pair_id[0] == "P") else "negative"
                    )
                    au_r_values["child_id"] = child
                    df_concat = pd.concat([df_concat, au_r_values]).reset_index(drop=True)
            df_concat.to_csv(extracted_data_savepath, index=False)

            print(f'Extracted feature csv {extracted_data_savepath} saved successfully!\n')
            if timestamp:
                return df_concat[features+['pair_id','round','child_id','condition', 'timestamp']]
            else:
                return df_concat[features+['pair_id','round','child_id','condition']]
    else:
        extracted_data_savepath = DATA_SAVEPATH / f"raw-extract-{summary_stat}-{position}.csv"
        df_concat = pd.DataFrame()

        if extracted_data_savepath.is_file():
            print('Data already exists in disk, reading..\n')
            df_concat = pd.read_csv(extracted_data_savepath, usecols=features+['pair_id','round','child_id','condition'])    
            return df_concat
        
        else:
            dummy_dat = pd.read_csv(csv_list[0])
            df_columns= ["pair_id","round","condition", "child_id"]+ [x for x in dummy_dat.columns if x not in ['frame','child_id','confidence','success', 'timestamp']] #list( [ 'AU07_r', 'AU06_r', 'AU10_r', 'AU20_r', 'AU25_r'] )#reading select activation units related columns from the dummy csv file - Inner brow raiser, Outer brow raiser, Cheek raiser,Lip stretcher
            print('Data does not exist in disk, creating a new extract csv..\n')
            # df_columns = ["pair_id", "round", "condition"] + list(
            #     ["AU20_c"]
            # )  # reading select activation units related columns from the dummy csv file - Lip stretcher

            df_concat = pd.DataFrame(columns=df_columns)  # mean activation units dataframe

            for csv in csv_list:
                pair_id = csv.stem.split("/")[-1].split("-")[2]
                condition = "positive" if (pair_id[0] == "P") else "negative"
                round = csv.stem.split("/")[-1].split("-")[4]
                data = pd.read_csv(csv)
                data = data.loc[data["success"] == 1]  # filtering
                child_ids =  data['child_id'].unique()
                for child in child_ids:
                    dat = data.loc[data['child_id'] == child].reset_index(drop = True)
                    au_values = dat[
                        [col for col in dat.columns if col not in ['frame','child_id','confidence','timestamp','success']] #
                    ].astype(float).reset_index(drop=True)
                    
                    df_concat.loc[ len(df_concat) ] = [pair_id, round, condition, child] + au_values.describe().loc[summary_stat].to_list()
            
            df_concat.to_csv(extracted_data_savepath, index=False)

            print(f'Extracted feature csv {extracted_data_savepath} saved successfully!\n')

            return df_concat[features+['pair_id','round','child_id','condition']]

def process_data(data, features, split_ratio):

    pair_id_list = list(set(data["pair_id"]))  # get unique pair_ids

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
    scaler.fit(data[features])

    x_train = data.loc[
        data["pair_id"].isin(pair_id_list[: split_index])
    ].reset_index(drop=True)

    x_train = scaler.transform(x_train[features])
    y_train = data.loc[
        data["pair_id"].isin(pair_id_list[: split_index])
    ].reset_index(drop=True)
    y_train = y_train["condition"].apply(lambda x: 1.0 if (x == "positive") else 0.0)

    x_test = data.loc[
        data["pair_id"].isin(pair_id_list[split_index :])
    ].reset_index(drop=True)
    x_test = scaler.transform(x_test[features])

    y_test = data.loc[
        data["pair_id"].isin(pair_id_list[split_index :])
    ].reset_index(drop=True)
    y_test = y_test["condition"].apply(lambda x: 1.0 if (x == "positive") else 0.0)
    
    
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    return x_train, x_test, y_train, y_test, split_stats

# if __name__ == "__main__":
#     X_train, X_test, Y_train, Y_test = process_data(0.7)
#     print(X_train.head())
#     print(X_train.tail())
#     print('-------------------------------------------------------------------------')
#     print(X_test.head())
#     print(X_test.tail())
#     print('-------------------------------------------------------------------------')
#     print(len(Y_train))
#     print('-------------------------------------------------------------------------')
#     print(len(Y_test))
