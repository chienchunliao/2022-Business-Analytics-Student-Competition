# -*- coding: utf-8 -*-
import pandas as pd, copy, time
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, f1_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.calibration import CalibratedClassifierCV

import warnings
warnings.filterwarnings('ignore')

def data_balancing(df, target_cloumn_name, method, random_state=20):
    """
    method= 'up' : oversampling
            'down': downsampling
            'combine': combing oversampling and downsampling
    """
    from sklearn.utils import resample
    # Determine the major and the minor dataset
    df_maj = df[df.loc[:, target_cloumn_name] == 0]
    df_min = df[df.loc[:, target_cloumn_name] == 1]
    if df_maj.shape[0] > df_min.shape[0]:
        pass
    else:
        df_maj, df_min = df_min, df_maj
    if method == 'up':
        df_min_upsamp = resample(df_min, 
                                 replace=True,     
                                 n_samples=df_maj.shape[0],
                                 random_state=random_state)
        df_ret = df_maj.append(df_min_upsamp)
    elif method == 'down':
        df_maj_dowsamp = resample(df_maj,
                                  replace=False,
                                  n_samples=df_min.shape[0],
                                  random_state=random_state)
        df_ret = df_maj_dowsamp.append(df_min)
    elif method == 'combine':
        avg_weight = (df_maj.shape[0] / df_min.shape[0]) ** 0.5
        df_min_upsamp = resample(df_min, 
                                 replace=True,     
                                 n_samples=round(df_min.shape[0]*avg_weight),
                                 random_state=random_state)
        df_maj_dowsamp = resample(df_maj, 
                                  replace=False,     
                                  n_samples=round(df_maj.shape[0]/avg_weight),
                                  random_state=random_state)
        df_ret = df_maj_dowsamp.append(df_min_upsamp)
    return df_ret

#%% Form All Possible Data Balancing Combinations to be trained
df_rese_dic = {'up':{},
               'down':{},
               'combine':{}}
df_rese_test_dic = {}

for file in os.listdir():
    if file.split('.')[-1] == 'csv':
        if file.split('_')[0] == 'Training':
            df = pd.read_csv(file)
            df = df.drop(['account_id'], axis=1)
            temp = file + 'resamp'
            for method in ['up', 'down', 'combine']:
                df_rese_dic[method][temp] = data_balancing(df, 'purchase_event', method)
        elif file.split('_')[0] == 'Testing':
            df = pd.read_csv(file)
            df = df.drop(['account_id'], axis=1)
            temp = file + 'resamp'
            df_rese_test_dic[temp] = df
        else:
            pass

#%% General Setting for Model Training 
n = -1
cv = 5
random = 20

#%% Training all data with all alogrithms
model_dic = {'KNN Classifier': 
             [KNeighborsClassifier(n_jobs=n), 
                       {'n_neighbors': range(1, 26)
                        }
                       ], 
             'Gradient Boosting Classifier': 
                 [GradientBoostingClassifier(random_state=random), 
                  {"loss": ['log_loss', 'deviance', 'exponential'], 
                   "n_estimators": [100, 200, 300, 400, 500],
                   "learning_rate":[0.2, 0.4, 0.6, 0.8, 1], 
                   "max_depth": range(3,11)
                   }
                  ], 
             'Linear Support Vector Classifier': 
                 [LinearSVC(random_state=random), 
                  {'penalty': ['l1', 'l2'],
                   'loss': ['hinge', 'squared_hinge'],
                   'C': [0.001, 0.01, 0.1, 1, 10, 100, 100000]
                   }
                  ], 
             'Support Vector Classifier': 
                 [SVC(probability=True, random_state=random), 
                  {'kernel': ['poly', 'rbf', 'sigmoid', 'precomputed'],
                   'C': [0.001, 0.01, 0.1, 1, 10, 100, 10000],
                   'gamma': [0.0001, 0.001,0.001,0.1,1,10, 100],
                   'degree': range(3, 7)
                   }
                  ], 
             'Random Forest Classifier': 
                 [RandomForestClassifier(n_jobs=n, random_state=random), 
                  {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, None], 
                   'max_features': ['auto', 'log2'],
                   'n_estimators': [20,50,100,150,200,300]
                   }
                  ],
             'Logistic Regression': 
                 [LogisticRegression(multi_class="auto", random_state=random, n_jobs=n),
                  {'penalty': ['l1', 'l2', 'elasticnet'], 
                   'C':[0.001, .009, 0.01, .09, 1, 5, 10, 25], 
                   'solver': ['liblinear', 'newton-cg', 'sag', 'saga', 'lbfgs'], 
                   'max_iter': [100, 1000, 2000]
                   }
                  ],
             'Naive Bayes Classifier': 
                 [CategoricalNB(), 
                  {'alpha': [0, 0.2, 0.4, 0.6, 0.8, 1]
                   }
                  ]
              }


result_lis = []
error_lis = []
inde = ['procssing method', 'balancing method', 'model name', 'best parameters', 'x train', 'y train', 'x test', 'y test', 'y test predicted', 'y test predicted probability', 'accuracy', 'precision', 'AUC', 'F1', 'time used(s)', 'best model']
for method_balance in df_rese_dic.keys():
    for method_processing in df_rese_dic[method_balance].keys():
        for model_name in model_dic:
            model = copy.deepcopy(model_dic[model_name][0])
            param_dic = model_dic[model_name][1]
            df = df_rese_dic[method_balance][method_processing].iloc[:51,:]
            x = df.drop(['purchase_event', 'number_employees_unknown', 'industry_unknow'], 
                        axis=1, 
                        errors='ignore')
            y = df['purchase_event']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=random)
            t1 = time.time()
            model_cv = GridSearchCV(model, param_dic, cv=cv, n_jobs=n)
            model_cv.fit(x_train, y_train)
            error_lis.append([model_name, x_train, y_train])

            best_param = model_cv.best_params_
            best_model = model_cv.best_estimator_
            pred = pd.Series(best_model.predict(x_test))
            if model_name == 'LinearSVC':
                clf = CalibratedClassifierCV(best_model, n_jobs=n) 
                clf.fit(x_train, y_train)
                y_test_pred_prob = clf.predict_proba(x_test)
            else:
                y_test_pred_prob = best_model.predict_proba(x_test)
            test_score_accuracy = accuracy_score(y_test, pred)
            test_score_precision = precision_score(y_test, pred)
            test_score_auc = roc_auc_score(y_test, pred)
            test_score_f1 = f1_score(y_test, pred)
            t2 = time.time()
            time_use = t2-t1
            se_temp = pd.Series([method_processing, method_balance, model_name, best_param, x_train, y_train, x_test, y_test, pred, y_test_pred_prob, test_score_accuracy, test_score_precision, test_score_auc, test_score_f1, time_use, best_model], 
                                index = inde)
            result_lis.append(se_temp)
            
df_result = pd.DataFrame(result_lis, columns=inde)

#%% Top five model score result & Final Model
criteria = 'AUC'
top_five_model_result = df_result.sort_values(criteria, ascending=False).iloc[:5, :]['model name', 'AUC']
final = df_result.sort_values(criteria, ascending=False).iloc[0, :]
final_model = final['best model']
 
y_test = final['y test']
y_pred_proba = final['y test predicted probability'][:,1]

#%% Generate ROC chart and AUC score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
fpr, tpr , threshold = roc_curve(y_test, y_pred_proba)
AUC_score = final['AUC']

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

#%% Generate Precision-Recall Curve
from sklearn.metrics import precision_recall_curve
precision, recall, threshold = precision_recall_curve(y_test, y_pred_proba)

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

#%% Generate Gain Chart
import scikitplot as skplt
skplt.metrics.plot_cumulative_gain(y_test, y_pred_proba)
plt.xlabel("Percentage of sample")
plt.ylabel("Gain")
plt.show()

#%% Generate predicted probability for the unknown test set
test_name = 'Testing_' + final['procssing method'].split('_',maxsplit=1)[1]
x_pred = df_rese_test_dic[test_name].drop(['purchase_event', 'number_employees_unknown', 'industry_unknow'], axis=1, errors='ignore')
if final['model name'] == 'LinearSVC':
    clf = CalibratedClassifierCV(final_model, n_jobs=n) 
    clf.fit(final['x train'], final['y train'])
    pred_prob_unknown = clf.predict_proba(x_pred)
else: 
    pred_prob_unknown = final_model.predict_proba(x_pred)

