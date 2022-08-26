# -*- coding: utf-8 -*-

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("2_data_TRAINING.csv")
count = df['purchase_event'].value_counts()

count_retail, count_health, count_finance, count_infra, count_unknow = 0,0,0,0,0
for ind, row in df.iterrows():
    if row['industry_vertical_retail'] == 1:
        count_retail += 1
    elif row['industry_vertical_healthcare'] == 1:
        count_health += 1
    elif row['industry_vertical_finance'] == 1:
        count_finance += 1
    elif row['industry_vertical_infrastructure'] == 1:
        count_infra += 1
    else:
        count_unknow += 1
        
print('Industry Structure')
print('retail: ', count_retail)
print('health: ', count_health)
print('finance: ', count_finance)
print('infrastructure: ', count_infra)
print('unknown: ', count_unknow)
