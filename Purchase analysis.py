# -*- coding: utf-8 -*-

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Training_employee1_industry1.csv')

size_lis = ['number_employess_less_than 1000',
            'number_employees_1000_10K',
            'number_employees_10K_50K',
            'number_employees_greater_than_50K', 
            'number_employees_unknown']
indu_lis = ['industry_vertical_retail',
            'industry_vertical_healthcare',
            'industry_vertical_finance',
            'industry_vertical_infrastructure', 
            'industry_unknow']

df_self = pd.DataFrame(0, 
                       index=size_lis,
                       columns=indu_lis)
df_com_1 = pd.DataFrame(0, 
                        index=size_lis,
                        columns=indu_lis)
df_com_2 = pd.DataFrame(0, 
                        index=size_lis,
                        columns=indu_lis)
df_com_3 = pd.DataFrame(0, 
                        index=size_lis,
                        columns=indu_lis)

for row_num in range(df.shape[0]):
    row = df.loc[row_num]
    
    if row['purchase_event'] == 1:
        for inde in size_lis:
            for col in indu_lis:
                num_self = row[inde] + row[col]
                if num_self > 1:
                    df_self.at[inde, col] += (num_self - 1) 
                
    elif sum([row['competitor_1_cloud'], row['competitor_1_networking']]) > 0:
        for inde in size_lis:
            for col in indu_lis:
                num_1 = row[inde] + row[col]
                if num_1 > 1:
                    df_com_1.at[inde, col] += (num_1 - 1) 
        
    elif sum([row['competitor_2_cloud'], row['competitor_2_networking']]) > 0:
        for inde in size_lis:
            for col in indu_lis:
                num_2 = row[inde] + row[col]
                if num_2 > 1:
                    df_com_2.at[inde, col] += (num_2 - 1) 
        
    elif sum([row['competitor_3_cloud'], row['competitor_3_networking']]) > 0:
        for inde in size_lis:
            for col in indu_lis:
                num_3 = row[inde] + row[col]
                if num_3 > 1:
                    df_com_3.at[inde, col] += (num_3 - 1) 
                    
df_comb = df_self.append(df_com_1)
df_comb = df_comb.append(df_com_2)
df_comb = df_comb.append(df_com_3)

#df_comb.to_csv('purchase_analysis.csv')