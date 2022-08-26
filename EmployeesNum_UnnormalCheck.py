# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('2_data_TRAINING.csv')
data_1 = data[['number_employees_greater_than_50K', 
                                      'number_employees_10K_50K', 
                                      'number_employees_1000_10K', 
                                      'number_employess_less_than 1000']]
name = np.array(data.columns)
count_emp_0 = 0
count_emp_1 = 0
count_emp_2 = 0
count_emp_3 = 0
count_emp_4 = 0
count_emp_err = 0
for row_num in range(data.shape[0]):
    emp_lis = list(data.loc[row_num][['number_employees_greater_than_50K', 
                                      'number_employees_10K_50K', 
                                      'number_employees_1000_10K', 
                                      'number_employess_less_than 1000']])
    if sum(emp_lis) == 0:
        count_emp_0 = count_emp_0 + 1
    elif sum(emp_lis) == 1:
        count_emp_1 = count_emp_1 + 1
    elif sum(emp_lis) == 2:
        count_emp_2 = count_emp_2 + 1
    elif sum(emp_lis) == 3:
        count_emp_3 = count_emp_3 + 1
    elif sum(emp_lis) == 4:
        count_emp_4 = count_emp_4 + 1
    else:
        count_emp_err = count_emp_err + 1
total = count_emp_0 + count_emp_1 + count_emp_2 + count_emp_3 + count_emp_4
print('employee size')
print("0 : ", count_emp_0)#/total)
print("1 : ", count_emp_1)#/total)
print("2 : ", count_emp_2)#/total)
print("3 : ", count_emp_3)#/total)
print("4 : ", count_emp_4)#/total)
print("error : ", count_emp_err)#/total)
