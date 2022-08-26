# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('2_data_TRAINING.csv')
name = np.array(data.columns)

nume_list = ['number_contacts', 'email_acitivity', 'it_spend_networking', 'it_spend_cloud', 'it_spend_others']
bina_list = []
for i in name:
    if (i not in nume_list) and (i != 'account_id'):
        bina_list.append(i)

print("number of outliers")
for col in nume_list:
    q1 = np.percentile(data[col], 25, interpolation = 'midpoint')
    q3 = np.percentile(data[col], 75, interpolation = 'midpoint')
    iqr = q3 - q1
    count_out = 0
    for cell in data[col]:
        if cell != 0:
            if (cell >= q3 + 1.5*iqr) or (cell <= q1 - 1.5*iqr):
                count_out = count_out + 1
    print(col, ':',count_out)
    
    