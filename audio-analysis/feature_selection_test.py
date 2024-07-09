import os
from pathlib import Path
import pandas as pd


from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

from utils import FEATURE_SETS, extract_raw_data

featureset_names = ['GeM', 'eGe', 'emob']
summary_stat = None #'mean'

for featureset_name in featureset_names:

    print(f'Featureset name: {featureset_name}, Summary stat: {summary_stat}')
    extractor = FEATURE_SETS[featureset_name]['extractor']
    features = FEATURE_SETS[featureset_name]['features'](extractor)

    aud_df = extract_raw_data(extractor, featureset_name, features, summary_stat)

    X = aud_df[[ x for x in ( set(aud_df.columns) - set(['pair_id','round', 'child_id', 'condition'] ) )  ]]
    y = aud_df['condition'].apply(lambda x: 0.0 if x == 'negative' else 1.0)

    est = sm.OLS(y, X)
    est2 = est.fit()
    print(est2.summary())


summary_stat = 'mean'
for featureset_name in featureset_names:

    print(f'Featureset name: {featureset_name}, Summary stat: {summary_stat}')
    extractor = FEATURE_SETS[featureset_name]['extractor']
    features = FEATURE_SETS[featureset_name]['features'](extractor)

    aud_df = extract_raw_data(extractor, featureset_name, features, summary_stat)

    X = aud_df[[ x for x in ( set(aud_df.columns) - set(['pair_id','round', 'child_id', 'condition'] ) )  ]]
    y = aud_df['condition'].apply(lambda x: 0.0 if x == 'negative' else 1.0)

    est = sm.OLS(y, X)
    est2 = est.fit()
    print(est2.summary())

# P-test, T-test and Chi-squared test