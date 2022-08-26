# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import numpy as np

data = pd.read_csv('2_data_TRAINING.csv')
name = np.array(data.columns)
count_ind_0 = 0
count_ind_1 = 0
count_ind_2 = 0
count_ind_3 = 0
count_ind_4 = 0
count_ind_err = 0

for row_num in range(data.shape[0]):
    ind_lis = list(data.loc[row_num][['industry_vertical_retail', 'industry_vertical_healthcare', 'industry_vertical_finance', 'industry_vertical_infrastructure']])
    if sum(ind_lis) == 0:
        count_ind_0 += 1
    elif sum(ind_lis) == 1:
        count_ind_1 += 1
    elif sum(ind_lis) == 2:
        count_ind_2 += 1
    elif sum(ind_lis) == 3:
        count_ind_3 += 1
    elif sum(ind_lis) == 4:
        count_ind_4 += 1
    else:
        count_ind_err += 1

print('Industry')
print("0 : ", count_ind_0)
print("1 : ", count_ind_1)
print("2 : ", count_ind_2)
print("3 : ", count_ind_3)
print("4 : ", count_ind_4)
print("error : ", count_ind_err)