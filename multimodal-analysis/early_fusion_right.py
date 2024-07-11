import os
import pandas as pd
from pathlib import Path
import logging
import datetime
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from bc_video.utils import extract_from_raw_data
from bc_audio.utils import FEATURE_SETS, generate_train_val_test_idx, calculate_random_baseline_from_splits, extract_raw_data, generate_train_val_split, generate_clean_train_and_val_idx, get_metrics, get_clean_foldwise_df

DIR_PATH = Path(__file__).parent

MODELS = {
        "Logistic_Regression": {'function': LogisticRegression(), 'param_grid': [ {'Logistic_Regression__solver': ['saga'], 'Logistic_Regression__penalty': ['l1','l2'],
                                                                                   'Logistic_Regression__tol': [1e-2, 1e-3, 1e-4, 1e-5]}]}, 
        
        "SVM": {'function': SVC(), 'param_grid': [{'SVM__kernel': ['poly'], 'SVM__degree': [2,3,4,5,6,7,8]},
                                                  {'SVM__kernel': ['linear', 'rbf', 'sigmoid']}]}, 
        
        "RF": {'function': RandomForestClassifier(), 'param_grid': [{'RF__n_estimators': [2,5,10,20,25],'RF__criterion': ['gini', 'entropy', 'log_loss'], 'RF__max_depth': [1,2,3,4,5,10,15,20,25,30],
                                                                                          'RF__max_features': [0.1, 0.2, 'sqrt', 'log2']}]},
        "KNN":{'function': KNeighborsClassifier(), 'param_grid': [{'KNN__n_neighbors': [1,2,3,4,5,7,9,11,15]}]}
    }
    
    #Solver sag supports only 'l2' or 'none' penalties
    #Solver newton-cholesky supports only 'l2' or 'none' penalties

VID_FEATURE_LIST = [
        ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y'],

        ['pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz'],

        ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'],

        ['AU06_r', 'AU12_r'],

        ['AU06_c', 'AU12_c'],

        ['AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']

]

AUD_FEATURE_SET =  [

 {'Name': 'emob', 'features' : [ 
                'mfcc_sma[1]',
                'mfcc_sma[2]',
                'mfcc_sma[3]',
                'mfcc_sma[4]',
                'mfcc_sma[5]',
                'mfcc_sma[6]',
                'mfcc_sma[7]',
                'mfcc_sma[8]',
                'mfcc_sma[9]',
                'mfcc_sma[10]',
                'mfcc_sma[11]',
                'mfcc_sma[12]',
               ]
    },
 {'Name': 'emob', 'features' : [ 
                'mfcc_sma[1]',
                'mfcc_sma[2]',
                'mfcc_sma[3]',
                'mfcc_sma[4]',
               ]
    },
 {'Name': 'emob', 'features' : [ 
                'mfcc_sma[5]',
                'mfcc_sma[6]',
                'mfcc_sma[7]',
                'mfcc_sma[8]',
               ]
    },
 {'Name': 'emob', 'features' : [ 
                'mfcc_sma[9]',
                'mfcc_sma[10]',
                'mfcc_sma[11]',
                'mfcc_sma[12]',
               ]
    },
    {'Name': 'emob', 'features' : [ 
                
                'pcm_loudness_sma',
               ]
    }, 
    {'Name': 'emob', 'features' : [ 
                'pcm_zcr_sma',
               ]
    }, 

    {'Name': 'emob', 'features' : [ 
                'F0env_sma'
               ]
    }, 
    {'Name': 'eGe', 'features' : [ 
                'F0semitoneFrom27.5Hz_sma3nz',
               ]
    },
]


SEED = 7

def main():

    experiment = 9

    cam_position = 'right'

    result_df = pd.DataFrame(columns = ['K_fold', 'Fold_number', 'Model', 'CV_Val_accuracy', 'CV_Val_RB', 'Test_accuracy_score', 'Test_precision_score', 'Test_recall_score'\
                                        , 'Test_F1_score', 'Test_Cohen_Kappa_score', 'Test_RB', 'Best_params', 'Aud_Features', 'Vid_Features', 'Featureset_number'])

    experiment_savepath = DIR_PATH / f'inference/early_fusion/experiment-{experiment}'

    if not os.path.exists(experiment_savepath):
        os.makedirs(experiment_savepath)    

    logging.basicConfig(filename= experiment_savepath / 'info_log.txt', filemode='a', format='%(message)s',datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

    dt = datetime.datetime.now()

    logging.info(dt.strftime("%c"))

    
    feat_counter = 0
    
    for vid_features in VID_FEATURE_LIST:
        
        vid_extracted_data = extract_from_raw_data(vid_features, summary_stat = 'mean', position=cam_position)

        for aud_featureset in AUD_FEATURE_SET:

            for K in range(3,9):
        
                logging.info(f"\n\n{K}-Fold Cross validation! \n")
    
                logging.info(f"Cam position: {cam_position} ! \n")

                extractor = FEATURE_SETS[aud_featureset['Name']]['extractor']
                #features = feature_set_items["features"](extractor)

                aud_features = aud_featureset['features'] #Custom features

                aud_mean_data = extract_raw_data(extractor, aud_featureset['Name'], aud_features, summary_stat = 'mean', custom_window_length_factor=4)
                aud_mean_data['round'] = aud_mean_data['round'].astype('int64')

                concat_data = pd.DataFrame()
                concat_data = pd.merge(vid_extracted_data, aud_mean_data, on=[x for x in vid_extracted_data.columns if x not in vid_features])

                print(concat_data.columns)

                combined_features = vid_features + aud_features

                logging.info(f'Featureset: {combined_features} , Number: {feat_counter}')

                _, val_idx, _, val_pair_ids = generate_train_val_split(concat_data, SEED, sklearn=True, k = K)

                for i in range(0,len(val_pair_ids)):
                
                    print(f'Fold - {i}/{K}')

                    logging.info(f'Fold - {i}/{K} \n')

                    val_pair_ids_kfold, val_idx_kfold = val_pair_ids.copy(), val_idx.copy()

                    test_pair_ids_kfold = val_pair_ids[i]

                    val_pair_ids_kfold.remove(test_pair_ids_kfold)

                    val_idx_kfold.remove(val_idx[i])

                    test_idx_kfold = concat_data.loc[ concat_data["pair_id"].isin(test_pair_ids_kfold)].index.to_list()

                    validation_sliced_df = concat_data.loc[ ~ concat_data.index.isin(test_idx_kfold) ].reset_index(drop=True)

                    train_idx_kfold, val_idx_kfold, train_pair_ids_kfold  = generate_clean_train_and_val_idx(validation_sliced_df, concat_data.pair_id.unique(), test_pair_ids_kfold, val_pair_ids_kfold) 
                    
                    # A demerit of using sklearn based train val splittng, some of the test indices bleed out and fill up rest of the train indices (As the split is Train - Test based)
                    
                    # Also, for some reason gridsearchcv seems to reset the index while performing the exhaustive search, therfore requiring a new set of index for a dataframe which does not contain test data to generate new validation and train indices

                    train_rb, val_rb, test_rb = calculate_random_baseline_from_splits(concat_data, train_pair_ids_kfold, val_pair_ids_kfold, test_pair_ids_kfold)

                    logging.info(f"Train pair IDs: {train_pair_ids_kfold}\n")

                    logging.info(f"Val pair IDs: {val_pair_ids_kfold}\n")

                    logging.info(f"Test pair IDs: {test_pair_ids_kfold}\n")

                    logging.info(f'Noting down the Random baseline metrics: \n')
                    
                    logging.info(f"Train : {train_rb}\n")

                    logging.info(f"Val : {val_rb}\n")

                    logging.info(f"Test : {test_rb}\n")

                    X_train = validation_sliced_df[combined_features]

                    Y_train = validation_sliced_df['condition'].apply(lambda x: 1.0 if (x == "positive") else 0.0)

                    X_test = concat_data.loc[ test_idx_kfold ][combined_features]

                    Y_test = concat_data.loc[ test_idx_kfold ]['condition'].apply(lambda x: 1.0 if (x == "positive") else 0.0)
                    
                    print('Data extracted successfully!  \n')

                    model_savepath = experiment_savepath / f'model/{K}-Fold/fold-{i}'
                    
                    if not os.path.exists(model_savepath):
                        os.makedirs(model_savepath)   

                    for name, items in MODELS.items():

                        print(f"Model: {name} ")        
                            
                        pipe = Pipeline([
                            ('sc', StandardScaler()),
                            (name, items['function'])
                        ])

                        best_model = GridSearchCV( estimator=pipe, param_grid= items["param_grid"], cv= zip(train_idx_kfold, val_idx_kfold) )

                        classifier = best_model.fit(X_train, Y_train)
                        
                        pred_test = best_model.predict(X_test)

                        test_acc, test_prec, test_rec, test_f1, test_ck = get_metrics(Y_test, pred_test)
                        
                        print(f'Model - {name} ; Best val score - {classifier.best_score_} ; Test score - {test_acc} ; Featureset number: {feat_counter} ; Fold - {K} \n')

                        logging.info(f'Model - {name} ; Best val score - {classifier.best_score_} ; Test score - {test_acc} ; Featureset number: {feat_counter} ; Fold - {K}')
                        
                        print('Saving model...')

                        model_savepath_with_filename = model_savepath / f"BM_{name}_featureset-{feat_counter}.joblib"

                        joblib.dump(best_model, model_savepath_with_filename, compress=0)
                        
                        print(f'Model saved successfully! @ {str(model_savepath_with_filename)}')
                        
                        result_df.loc[len(result_df)] = [K, i, name, classifier.best_score_, val_rb, test_acc, test_prec, test_rec, test_f1, \
                                                        test_ck, test_rb, classifier.best_params_, aud_features, vid_features, feat_counter]
            
            feat_counter += 1

    result_df.to_csv(experiment_savepath / 'results_df_test.csv', index=False)
    # result_df.sort_values(by=['Test_accuracy_score'], ascending=False).to_csv(experiment_savepath / 'results_sorted_acc.csv', index=False)
    # result_df.sort_values(by=['Test_Cohen_Kappa_score'], ascending=False).to_csv(experiment_savepath / 'results_sorted_ck.csv', index=False)

    clean_results_df = get_clean_foldwise_df(pd.read_csv(experiment_savepath / 'results_df_test.csv'), modality='Multi')
    clean_results_df.to_csv(experiment_savepath / 'clean_results_df_test.csv', index=False)

if __name__ == "__main__":
    main()
    # /media/chagan/2023-ELECTRA-field-MarcSchoolStudy/.venv/bin/python /media/chagan/2023-ELECTRA-field-MarcSchoolStudy/bc_video/binary_classifier_reworked.py
    # /media/chagan/2023-ELECTRA-field-MarcSchoolStudy/.venv/bin/python /media/chagan/2023-ELECTRA-field-MarcSchoolStudy/bc_audio/sklearn_binary_classifier.py
