import os
from pathlib import Path
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import datetime
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


from bc_audio.utils import MODELS, FEATURE_SETS, extract_raw_data, generate_train_val_split, calculate_random_baseline_from_splits, return_indices_as_list, generate_clean_train_and_val_idx, get_metrics, get_clean_foldwise_df


feat = [

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

    DIR_PATH = Path(__file__).parent

    experiment = 10

    result_df = pd.DataFrame(columns = ['K_fold', 'Fold_number', 'Model', 'CV_Val_accuracy', 'CV_Val_RB', 'Test_accuracy_score', 'Test_precision_score', 'Test_recall_score'\
                                        , 'Test_F1_score', 'Test_Cohen_Kappa_score', 'Test_MCC_score', 'Test_RB', 'Best_params', 'Features', 'Featureset_number'])

    experiment_savepath = DIR_PATH / f'inference/sklearn/mean/reworked/experiment-{experiment}'

    if not os.path.exists(experiment_savepath):
        os.makedirs(experiment_savepath)    

    logging.basicConfig(filename= experiment_savepath / 'info_log.txt', filemode='a', format='%(message)s',datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

    dt = datetime.datetime.now()

    logging.info(dt.strftime("%c"))

    feat_counter = 0 

    for featureset in feat:
        
        for K in [5]: #range(3,9)
            
            logging.info(f"\n\n{K}-Fold Cross validation! \n")

            logging.info(f"Feature set: {featureset['Name']}\n")
            
            print(f"Feature set: {featureset['features']}\n")

            extractor = FEATURE_SETS[featureset['Name']]['extractor']
            #features = feature_set_items["features"](extractor)
            
            features = featureset['features'] #Custom features
            
            aud_mean_data = extract_raw_data(extractor, featureset['Name'], features, summary_stat = 'mean')

            logging.info(f'Featureset: {features} , Number: {feat_counter}')

            _, val_idx, _, val_pair_ids = generate_train_val_split(aud_mean_data, SEED, sklearn=True, k = K)

            
            for i in range(0,len(val_pair_ids)):
                
                print(f'Fold - {i}/{K}')

                logging.info(f'Fold - {i}/{K} \n')

                val_pair_ids_kfold, val_idx_kfold = val_pair_ids.copy(), val_idx.copy()

                test_pair_ids_kfold = val_pair_ids[i]

                val_pair_ids_kfold.remove(test_pair_ids_kfold)

                val_idx_t = val_idx[i]

                val_idx_kfold.remove(val_idx_t)

                test_idx_kfold = aud_mean_data.loc[ aud_mean_data["pair_id"].isin(test_pair_ids_kfold)].index.to_list()

                validation_sliced_df = aud_mean_data.loc[ ~ aud_mean_data.index.isin(test_idx_kfold) ].reset_index(drop=True)

                train_idx_kfold, val_idx_kfold, train_pair_ids_kfold  = generate_clean_train_and_val_idx(validation_sliced_df, aud_mean_data.pair_id.unique(), test_pair_ids_kfold, val_pair_ids_kfold) 
                
                # A demerit of using sklearn based train val splittng, some of the test indices bleed out and fill up rest of the train indices (As the split is Train - Test based)

                # Also, for some reason gridsearchcv seems to reset the index while performing the exhaustive search, therfore requiring a new set of index for a dataframe which does not contain test data to generate new validation and train indices

                train_rb, val_rb, test_rb = calculate_random_baseline_from_splits(aud_mean_data, train_pair_ids_kfold, val_pair_ids_kfold, test_pair_ids_kfold)

                logging.info(f"Train pair IDs: {train_pair_ids_kfold}\n")

                logging.info(f"Val pair IDs: {val_pair_ids_kfold}\n")

                logging.info(f"Test pair IDs: {test_pair_ids_kfold}\n")

                logging.info(f'Noting down the Random baseline metrics: \n')
                
                logging.info(f"Train : {train_rb}\n")

                logging.info(f"Val : {val_rb}\n")

                logging.info(f"Test : {test_rb}\n")

                X_train = validation_sliced_df[features]

                Y_train = validation_sliced_df['condition'].apply(lambda x: 1.0 if (x == "positive") else 0.0) #aud_mean_data["pair_id"].isin([x for x in aud_mean_data['pair_id'].unique() if x not in test_pair_ids_kfold]) ~ aud_mean_data.index.isin(test_idx_kfold)

                X_test = aud_mean_data.loc[ test_idx_kfold ][features] # aud_mean_data["pair_id"].isin(test_pair_ids_kfold)

                Y_test = aud_mean_data.loc[ test_idx_kfold ]['condition'].apply(lambda x: 1.0 if (x == "positive") else 0.0) #aud_mean_data["pair_id"].isin(test_pair_ids_kfold)

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

                    test_acc, test_prec, test_rec, test_f1, test_ck, test_phi = get_metrics(Y_test, pred_test)
                    
                    print(f'Model - {name} ; Best val score - {classifier.best_score_}; Featureset number: {feat_counter} ; Fold - {K} \n')

                    print(f'Test metrics (Acc, F1, Cohen_Kappa, Prec, Recall): {test_acc}, {test_f1}, {test_ck}, {test_prec}, {test_rec} ')

                    logging.info(f'Model - {name} ; Best val score - {classifier.best_score_}; Featureset number: {feat_counter} ; Fold - {K} \n')

                    logging.info(f'Test metrics (Acc, F1, Cohen_Kappa, Prec, Recall, Phi): {test_acc}, {test_f1}, {test_ck}, {test_prec}, {test_rec}, {test_phi} ')
                    
                    print('Saving model...')

                    model_savepath_with_filename = model_savepath / f"BM_{name}_featureset-{feat_counter}.joblib"

                    joblib.dump(best_model, model_savepath_with_filename, compress=0)
                    
                    print(f'Model saved successfully! @ {str(model_savepath_with_filename)}')

                    result_df.loc[len(result_df)] = [K, i, name, classifier.best_score_, val_rb, test_acc, test_prec, test_rec, test_f1, \
                                                     test_ck, test_phi, test_rb, classifier.best_params_, features, feat_counter]

                train_pair_ids_kfold.clear()
                val_pair_ids_kfold.clear()
                train_idx_kfold.clear()
                val_idx_kfold.clear()

        feat_counter += 1

    result_df.to_csv(experiment_savepath / 'results_df_test.csv', index=False)
    # result_df.sort_values(by=['Test_accuracy_score'], ascending=False).to_csv(experiment_savepath / 'results_sorted_acc.csv', index=False)
    # result_df.sort_values(by=['Test_Cohen_Kappa_score'], ascending=False).to_csv(experiment_savepath / 'results_sorted_ck.csv', index=False)

    clean_results_df = get_clean_foldwise_df(pd.read_csv(experiment_savepath / 'results_df_test.csv'), modality='Uni')
    clean_results_df.to_csv(experiment_savepath / 'clean_results_df_test.csv', index=False)


if __name__ == '__main__':
    main()