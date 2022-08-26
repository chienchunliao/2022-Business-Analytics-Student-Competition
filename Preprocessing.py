# -*- coding: utf-8 -*-
import pandas as pd

#%% Import all data automatically
name_dic = {'Training': '2_data_TRAINING.csv', 
            'Testing': '3_data_TEST.csv'}
for typ, file_name in name_dic.items():
    df = pd.read_csv(file_name)
    col_name = df.columns
    for i in range(df.shape[0]):
        if (df.at[i,'persona_tech'] != 0) and (df.at[i, 'persona_tech'] != 1):
            df.at[i,'persona_tech'] = 1

    #%% Method 1
    ### Categorize records with problem in employee numbers columns to the unknown category
    df_emp_1 = df.copy()
    df_emp_1['number_employees_unknown'] = None
    for ind, row in df_emp_1.iterrows():
        temp = [row['number_employees_greater_than_50K'],
                row['number_employees_10K_50K'],
                row['number_employees_1000_10K'],
                row['number_employess_less_than 1000']]
        if sum(temp) == 1:
            df_emp_1.at[ind, 'number_employees_unknown'] = 0
        else:
            df_emp_1.at[ind, 'number_employees_greater_than_50K'] = 0
            df_emp_1.at[ind, 'number_employees_10K_50K'] = 0
            df_emp_1.at[ind, 'number_employees_1000_10K'] = 0
            df_emp_1.at[ind, 'number_employess_less_than 1000'] = 0
            df_emp_1.at[ind, 'number_employees_unknown'] = 1
    df_tem_1 = df_emp_1[['number_employees_greater_than_50K', 
                         'number_employees_10K_50K', 
                         'number_employees_1000_10K', 
                         'number_employess_less_than 1000', 
                         'number_employees_unknown']]
    
    #%% Method 2
    ### Drop records with problems in employee numbers columns
    df_emp_2 = df.copy()
    df_emp_2['number_employees_unknown'] = None
    for ind, row in df_emp_2.iterrows():
        temp = [row['number_employees_greater_than_50K'],
                row['number_employees_10K_50K'],
                row['number_employees_1000_10K'],
                row['number_employess_less_than 1000']]
        if sum(temp) == 1:
            df_emp_2.at[ind, 'number_employees_unknown'] = 0
        elif sum(temp) == 0:
            df_emp_2.at[ind, 'number_employees_unknown'] = 1
        else:
            df_emp_2.drop([ind])
    df_tem_2 = df_emp_2[['number_employees_greater_than_50K', 
                         'number_employees_10K_50K', 
                         'number_employees_1000_10K', 
                         'number_employess_less_than 1000', 
                         'number_employees_unknown']]
    
    #%% Method 3
    ### Drop all columns associated with employee numbers 
    df_emp_3 = df.copy()
    df_emp_3 = df_emp_3.drop(['number_employees_greater_than_50K', 
                              'number_employees_10K_50K', 
                              'number_employees_1000_10K', 
                              'number_employess_less_than 1000'], 
                              axis=1)
    
    #%% Process the columns associated with the industry vertical
    df_emp_list = [df_emp_1, df_emp_2, df_emp_3]
    count = 1
    for df_emp in df_emp_list:
        ### Categorize records with problem in industry vertical columns to the unknown category
        df_1 = df_emp.copy()
        df_1['industry_unknow'] = None
        for ind, row in df_1.iterrows():
            row = df_1.loc[ind]
            temp = [row['industry_vertical_retail'],
                    row['industry_vertical_healthcare'],
                    row['industry_vertical_finance'],
                    row['industry_vertical_infrastructure']]
            if sum(temp) == 1:
                df_1.at[ind,'industry_unknow'] = 0
            else:
                df_1.at[ind,'industry_vertical_retail'] = 0
                df_1.at[ind,'industry_vertical_healthcare'] = 0
                df_1.at[ind,'industry_vertical_finance'] = 0
                df_1.at[ind,'industry_vertical_infrastructure'] = 0
                df_1.at[ind,'industry_unknow'] = 1
        save_name_1 = typ + '_employee' + str(count) + '_industry1.csv'
        df_1.to_csv(save_name_1, index=False)
        
        ### Drop records with problems in industry vertical columns
        df_2 = df_emp.copy()
        df_2['industry_unknow'] = None
        for ind, row in df_2.iterrows():
            temp = [row['industry_vertical_retail'],
                    row['industry_vertical_healthcare'],
                    row['industry_vertical_finance'],
                    row['industry_vertical_infrastructure']]
            if sum(temp) == 1:
                df_2.at[ind,'industry_unknow'] = 0
            elif sum(temp) == 0:
                df_2.at[ind,'industry_unknow'] = 1
            else:
                df_2.drop([ind])
        save_name_2 = typ + '_employee' + str(count) + '_industry2.csv'
        df_2.to_csv(save_name_2, index=False)
        
        ### Drop all columns associated with industry vertical
        df_3 = df_emp.copy()
        df_3.drop(['industry_vertical_retail', 
                   'industry_vertical_healthcare', 
                   'industry_vertical_finance', 
                   'industry_vertical_infrastructure'],axis=1)
        save_name_3 = typ + '_employee' + str(count) + '_industry3.csv'
        df_3.to_csv(save_name_3, index=False)
        
        count += 1