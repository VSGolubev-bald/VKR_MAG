import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import base64
import array
import itertools
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

ALL_SIGNAL_DERIVATIONS = ["I",   "II", "III",  "aVR", "aVL", "aVF", "V1",  "V2", "V3",   "V4",  "V5",  "V6"]
ALL_SIGNAL_DERIVATIONS = ['std1', 'std2', 'std3',  "aVR", "aVL", "aVF", "V1",  "V2", "V3",   "V4",  "V5",  "V6"]
SIGNAL_DERIVATIONS_AVG = "УКЦ_Группа"
SIGNAL_DERIVATIONS_REPR = "РКЦ"
AVG_MAX_GROUPS_FIELD_NAME = "Количество групп"
REPR_MAX_GROUPS_FIELD_NAME = "Количество РКЦ"
UNITS_IN_MV_FIELD_NAME = "Вес бита, мкВ"
SAMPLE_RATE_FIELD_NAME = "Частота оцифровки"
START_POS_FIELD_NAME = 'Позиция от начала сигнала'
MAX_AVG_GROUPS = 4
MAX_REPR_GROUPS = 20

GROUPS_AVG = [i + 1 for i in range(4)]
GROUPS_REPR = [i + 1 for i in range(20)]

# Блок P, Q, R, S, T
BASE_PARAMS_HR_FILEDS_PREFIX = ['ЧСС_']
BASE_PARAMS_QRS_FILEDS_PREFIX = ['QRS, мс_', 'Q_', 'R_', 'S_']
BASE_PARAMS_QT_FILEDS_PREFIX = ['QT_', 'QTcH_', 'QTcB_', 'QTcF_', 'QTcL_']
BASE_PARAMS_P_FILEDS_PREFIX = ['P, мс_', 'P начало_', 'P конец_']
BASE_PARAMS_T_FILEDS_PREFIX = ['T, мс_', 'T начало_', 'T конец_']
BASE_PARAMS_PQ_FILEDS_PREFIX = ['PQ, мс_']

# Блок Оси
BASE_PARAMS_AXES_FILEDS_PREFIX = ['Ось QRS_', 'Ось P_', 'Ось T_']
BASE_PARAMS_RR_ARRAY_FILEDS_PREFIX = 'RR интервалы, мс_'
BASE_PARAMS_RR_ARRAY_FILEDS_PREFIX_1 = BASE_PARAMS_RR_ARRAY_FILEDS_PREFIX + SIGNAL_DERIVATIONS_AVG + "_1"

# Блок R-R
BASE_PARAMS_RR_FILEDS_PREFIX = ['RR мин, мс_', 'RR макс, мс_', 'RR средн, мс_',
                                'RR дисперсия_', 'RR медиана_',
                                'RR DMinMaxDivMean_',
                                'RR всего КЦ, шт_', 'RR сигма_', 'RR MSE_',
                                BASE_PARAMS_RR_ARRAY_FILEDS_PREFIX, 'RR интервалы, шт_', ]

BASE_PARAMS_FILEDS_PREFIX = []
BASE_PARAMS_FILEDS_PREFIX.append(BASE_PARAMS_HR_FILEDS_PREFIX)
BASE_PARAMS_FILEDS_PREFIX.append(BASE_PARAMS_QRS_FILEDS_PREFIX)
BASE_PARAMS_FILEDS_PREFIX.append(BASE_PARAMS_QT_FILEDS_PREFIX)
BASE_PARAMS_FILEDS_PREFIX.append(BASE_PARAMS_P_FILEDS_PREFIX)
BASE_PARAMS_FILEDS_PREFIX.append(BASE_PARAMS_T_FILEDS_PREFIX)
BASE_PARAMS_FILEDS_PREFIX.append(BASE_PARAMS_PQ_FILEDS_PREFIX)
BASE_PARAMS_FILEDS_PREFIX.append(BASE_PARAMS_AXES_FILEDS_PREFIX)
BASE_PARAMS_FILEDS_PREFIX.append(BASE_PARAMS_RR_FILEDS_PREFIX)
BASE_PARAMS_FILEDS_PREFIX_FLATTEN_LIST = list(itertools.chain(*BASE_PARAMS_FILEDS_PREFIX))

names_avg = [[]]
names_avg = [[chan + '_' + SIGNAL_DERIVATIONS_AVG + '_' + str(j) for chan, n in zip(ALL_SIGNAL_DERIVATIONS, range(1, len(ALL_SIGNAL_DERIVATIONS) + 1))]
         for j in GROUPS_AVG]

names_repr = [[]]
names_repr = [[chan + '_' + SIGNAL_DERIVATIONS_REPR + '_' + str(j) for chan, n in zip(ALL_SIGNAL_DERIVATIONS, range(1, len(ALL_SIGNAL_DERIVATIONS) + 1))]
         for j in GROUPS_REPR]

names = names_avg + names_repr

DERIV_PARAMS_FILEDS_PREFIX = 'Отв '

# Блок "P (P,P',амплитуда,длина)
DERIV_PARAMS_FILEDS_PREFIX_P = [
                                'P, мс',
                                'P, мВ',
                                "P', мс",
                                "P', мВ",
                                'P начало',
                                'P конец',
                                "P' начало",
                                "P' конец"
                                ]

# Блок "QRS (тип,Q,R,S,R',S',амплитуда,длина)"
DERIV_PARAMS_FILEDS_PREFIX_QRS = [
                                  "Q тип", "Q, мс", "Q, мВ",
                                  "R, мс", "R, мВ",
                                  "S, мс", "S, мВ",
                                  "R', мс", "R', мВ",
                                  "S', мс", "S', мВ" ,
                                  'Q начало',
                                  'Q конец',
                                  'R начало',
                                  'R конец',
                                  'S начало',
                                  'S конец',
                                  "R' начало",
                                  "R' конец",
                                  "S' начало",
                                  "S' конец"
                                  ]

# Блок "T (T, T', амплитуда, длина)"
DERIV_PARAMS_FILEDS_PREFIX_T = [
                                "T, мс",
                                "T, мВ",
                                "T', мс",
                                "T', мВ",
                                'T начало',
                                'T конец',
                                "T' начало",
                                "T' конец"
                                ]

# Блок "ST (STj, STj80, наклон)"
DERIV_PARAMS_FILEDS_PREFIX_ST = [
                                 "STj, мВ",
                                 "STj80, мВ",
                                 "ST наклон, мВ/сек"
                                 ]

DERIV_PARAMS_FILEDS_PREFIX_P = [ DERIV_PARAMS_FILEDS_PREFIX + i + '_' for i in DERIV_PARAMS_FILEDS_PREFIX_P ]
DERIV_PARAMS_FILEDS_PREFIX_QRS = [ DERIV_PARAMS_FILEDS_PREFIX + i + '_' for i in DERIV_PARAMS_FILEDS_PREFIX_QRS ]
DERIV_PARAMS_FILEDS_PREFIX_T = [ DERIV_PARAMS_FILEDS_PREFIX + i + '_' for i in DERIV_PARAMS_FILEDS_PREFIX_T ]
DERIV_PARAMS_FILEDS_PREFIX_ST = [ DERIV_PARAMS_FILEDS_PREFIX + i + '_' for i in DERIV_PARAMS_FILEDS_PREFIX_ST ]

DERIV_PARAMS_FILEDS_PREFIX = []
DERIV_PARAMS_FILEDS_PREFIX.append(DERIV_PARAMS_FILEDS_PREFIX_P)
DERIV_PARAMS_FILEDS_PREFIX.append(DERIV_PARAMS_FILEDS_PREFIX_QRS)
DERIV_PARAMS_FILEDS_PREFIX.append(DERIV_PARAMS_FILEDS_PREFIX_T)
DERIV_PARAMS_FILEDS_PREFIX.append(DERIV_PARAMS_FILEDS_PREFIX_ST)
DERIV_PARAMS_FILEDS_PREFIX_FLATTEN_LIST = list(itertools.chain(*DERIV_PARAMS_FILEDS_PREFIX))

# -----------------------------------------------------------------------
#
# -----------------------------------------------------------------------
def getDerivParamName(param, derivation, group, average=True):
  return param + (SIGNAL_DERIVATIONS_AVG if average else SIGNAL_DERIVATIONS_REPR) + '_' + str(group + 1) + '_' + derivation


names_deriv_avg = [[]]
names_deriv_avg = [[getDerivParamName(param, chan, j, True)
                    # param + SIGNAL_DERIVATIONS_AVG + '_' + str(j) + '_' + chan
                    for chan, n in zip(ALL_SIGNAL_DERIVATIONS, range(1, len(ALL_SIGNAL_DERIVATIONS) + 1))]
         for j in GROUPS_AVG for param in DERIV_PARAMS_FILEDS_PREFIX_FLATTEN_LIST]

names_deriv_repr = [[]]
names_deriv_repr = [[getDerivParamName(param, chan, j, False)
                    #  param + SIGNAL_DERIVATIONS_REPR + '_' + str(j) + '_' + chan
                     for chan, n in zip(ALL_SIGNAL_DERIVATIONS, range(1, len(ALL_SIGNAL_DERIVATIONS) + 1))]
         for j in GROUPS_REPR for param in DERIV_PARAMS_FILEDS_PREFIX_FLATTEN_LIST]

names_deriv_params = names_deriv_avg + names_deriv_repr


# -----------------------------------------------------------------------
# Получение количества групп для УКЦ (для заданной ЭКГ)
# -----------------------------------------------------------------------
def getAvgGroupsCountForECG(df, number_of_ecg = 0):
  groups = MAX_AVG_GROUPS
  if AVG_MAX_GROUPS_FIELD_NAME in df.columns:
    groups = int(df[AVG_MAX_GROUPS_FIELD_NAME][number_of_ecg])
  return groups

# -----------------------------------------------------------------------
# Получение количества КЦ для РКЦ (для заданной ЭКГ)
# -----------------------------------------------------------------------
def getReprGroupsCountForECG(df, number_of_ecg = 0):
  groups = MAX_REPR_GROUPS
  if REPR_MAX_GROUPS_FIELD_NAME in df.columns:
    groups = int(df[REPR_MAX_GROUPS_FIELD_NAME][number_of_ecg])
  return groups

# -----------------------------------------------------------------------
#
# -----------------------------------------------------------------------
'''def base64ToInt(x):
  # 'h' signed short
  return np.array(array.array('h', base64.b64decode(x))) if len(str(x)) > 0 and str(x) != 'nan' else x '''

def base64ToInt(x):
# 'h' signed short

  if isinstance(x, str) and len(str(x)) > 0 and str(x) != 'nan':
    try:
      base64.b64decode(x)
    except:
      x = x + "=="
    parsed_base64 = base64.b64decode(x)
    if len(parsed_base64) % 2 != 0:
      parsed_base64 = parsed_base64[:-1]
    return np.array(array.array('h', parsed_base64))
  else:
    return x


# -----------------------------------------------------------------------
# Смещение от начала сигнала для заданного РКЦ (в векторах)
# -----------------------------------------------------------------------
def getStartPosFieldForGroup(group):
  return START_POS_FIELD_NAME + '_' + SIGNAL_DERIVATIONS_REPR + '_' + str(group + 1)


names_ecg = ["std1_УКЦ_Группа_","std2_УКЦ_Группа_","std3_УКЦ_Группа_",
               "aVR_УКЦ_Группа_","aVL_УКЦ_Группа_","aVF_УКЦ_Группа_",
              "V1_УКЦ_Группа_","V2_УКЦ_Группа_","V3_УКЦ_Группа_","V4_УКЦ_Группа_",
              "V5_УКЦ_Группа_","V6_УКЦ_Группа_"]

names_middle = []
names_min = []
names_max = []

# QTcB, Ось QRS ???
'''
Accuracy           : 0.8890449438202247
F1-Score           : 0.5212121212121212
False Positive Rate: 0.11544227886056972
False Negative Rate: 0.044444444444444446
Precision          : 0.35833333333333334
Recall             : 0.9555555555555556
Specificity        : 0.8845577211394303
tn                 : 590 667 88.46%
tp                 : 43 45 95.56%
fp (ошибка 1 рода) : 77 667 11.54%
fn (ошибка 2 рода) : 2 45 4.44%
================================'''

names_min.append(["label", "Ось QRS","ЧСС","QRS, мс","Фронтальный угол","CenterMass","QT","QTcB"]) # ,"QTcH","QTcB","QTcL","Ось QRS","P начало","P конец"

names_middle.append(["label", 'ЧСС', 'QRS, мс', 'Фронтальный угол', 'CenterMass', 'Ось QRS',
       'QT', 'QTcH', 'QTcB', 'QTcL', 'Возраст', 'R', 'S', 'QTcF', 'T начало',
       'T конец', 'T, мс', 'Ось T',
       'Корнельский показатель',
       'Индекс Соколова–Лайона', 'RaVL, мВ'])
'''
names_middle.append(["label", "ЧСС","QRS, мс","Фронтальный угол","CenterMass","Ось QRS","QT","QTcH","QTcB","QTcL","Возраст"]) #
names_middle.append(["R","S","Q","QTcF","T начало","T конец","T, мс","Ось T","P начало","P конец"])
names_middle.append(["Корнельское произведение", "Корнельский показатель", "Индекс Соколова–Лайона", "RI, мВ", "RaVL, мВ", "RI+SIII, мВ",
                     "PQ, мс", "Ось P", 'RR мин, мс', 'RR макс, мс', 'RR средн, мс', 'RR MSE', 'RR медиана', 'RR интервалы, шт'])
'''
names_max.append(['Возраст', 'Корнельский показатель', 'Индекс Соколова–Лайона', 'Rstd1, мВ', 'RaVL, мВ', 'Rstd1+Sstd3, мВ', 'ЧСС', 'QRS, мс',
                   'QT', 'QTcH', 'QTcB', 'QTcF', 'QTcL', 'QRS начало', 'R', 'CenterMass', 'S', 'T начало', 'T конец', 'T, мс', 'Ось QRS',
                   'Фронтальный угол', 'Ось T', 'Частота оцифровки', 'Вес бита, мкВ', 'Количество РКЦ', 'Количество групп', 'ЧСС_УКЦ_Группа_1',
                   'QRS, мс_УКЦ_Группа_1', 'QT_УКЦ_Группа_1', 'QTcH_УКЦ_Группа_1', 'QTcB_УКЦ_Группа_1', 'QTcF_УКЦ_Группа_1', 'QTcL_УКЦ_Группа_1',
                   'QRS начало_УКЦ_Группа_1', 'R_УКЦ_Группа_1', 'S_УКЦ_Группа_1', 'T начало_УКЦ_Группа_1', 'T конец_УКЦ_Группа_1', 'T, мс_УКЦ_Группа_1',
                   'CenterMass_УКЦ_Группа_1', 'Ось QRS_УКЦ_Группа_1', 'Ось T_УКЦ_Группа_1', 'Корнельское произведение_УКЦ_Группа_1',
                   'Корнельский показатель_УКЦ_Группа_1', 'Индекс Соколова–Лайона_УКЦ_Группа_1', 'Rstd1, мВ_УКЦ_Группа_1', 'RaVL, мВ_УКЦ_Группа_1',
                   'Rstd1+Sstd3, мВ_УКЦ_Группа_1', 'Отв R, мс_УКЦ_Группа_1_std1', 'Отв R, мс_УКЦ_Группа_1_std2', 'Отв R, мс_УКЦ_Группа_1_aVL',
                   'Отв R, мс_УКЦ_Группа_1_aVF', 'Отв R, мс_УКЦ_Группа_1_V3', 'Отв R, мс_УКЦ_Группа_1_V4', 'Отв R, мс_УКЦ_Группа_1_V5',
                   'Отв R, мс_УКЦ_Группа_1_V6', 'Отв S, мс_УКЦ_Группа_1_V4', 'Отв R начало_УКЦ_Группа_1_std1', 'Отв R начало_УКЦ_Группа_1_std2',
                   'Отв R начало_УКЦ_Группа_1_aVL', 'Отв R начало_УКЦ_Группа_1_aVF', 'Отв R начало_УКЦ_Группа_1_V3', 'Отв R начало_УКЦ_Группа_1_V4',
                   'Отв R начало_УКЦ_Группа_1_V5', 'Отв R начало_УКЦ_Группа_1_V6', 'Отв R конец_УКЦ_Группа_1_std1', 'Отв R конец_УКЦ_Группа_1_std2',
                   'Отв R конец_УКЦ_Группа_1_aVL', 'Отв R конец_УКЦ_Группа_1_aVF', 'Отв R конец_УКЦ_Группа_1_V3', 'Отв R конец_УКЦ_Группа_1_V4',
                   'Отв R конец_УКЦ_Группа_1_V5', 'Отв R конец_УКЦ_Группа_1_V6', 'Отв S начало_УКЦ_Группа_1_V4', 'Отв S конец_УКЦ_Группа_1_V4',
                   'Отв T, мс_УКЦ_Группа_1_std1', 'Отв T, мс_УКЦ_Группа_1_std2', 'Отв T, мс_УКЦ_Группа_1_aVR', 'Отв T, мс_УКЦ_Группа_1_V1',
                   'Отв T, мс_УКЦ_Группа_1_V2', 'Отв T, мс_УКЦ_Группа_1_V3', 'Отв T, мс_УКЦ_Группа_1_V4', 'Отв T, мс_УКЦ_Группа_1_V5',
                   'Отв T, мс_УКЦ_Группа_1_V6', 'Отв T начало_УКЦ_Группа_1_std1', 'Отв T начало_УКЦ_Группа_1_std2', 'Отв T начало_УКЦ_Группа_1_aVR',
                   'Отв T начало_УКЦ_Группа_1_V1', 'Отв T начало_УКЦ_Группа_1_V2', 'Отв T начало_УКЦ_Группа_1_V3', 'Отв T начало_УКЦ_Группа_1_V4',
                   'Отв T начало_УКЦ_Группа_1_V5', 'Отв T начало_УКЦ_Группа_1_V6', 'Отв T конец_УКЦ_Группа_1_std1', 'Отв T конец_УКЦ_Группа_1_std2',
                   'Отв T конец_УКЦ_Группа_1_aVR', 'Отв T конец_УКЦ_Группа_1_V1', 'Отв T конец_УКЦ_Группа_1_V2', 'Отв T конец_УКЦ_Группа_1_V3',
                   'Отв T конец_УКЦ_Группа_1_V4', 'Отв T конец_УКЦ_Группа_1_V5', 'Отв T конец_УКЦ_Группа_1_V6', 'Номер группы_РКЦ_1',
                   'Позиция от начала сигнала_РКЦ_1', 'Номер группы_РКЦ_2', 'Позиция от начала сигнала_РКЦ_2', 'Номер группы_РКЦ_3',
                   'Позиция от начала сигнала_РКЦ_3', 'Номер группы_РКЦ_4', 'Позиция от начала сигнала_РКЦ_4', 'Номер группы_РКЦ_5',
                   'Позиция от начала сигнала_РКЦ_5', 'Номер группы_РКЦ_6', 'Позиция от начала сигнала_РКЦ_6', 'label'])



'''
for i in names_deriv_avg:
  for l in i:
    if "Отв R, мВ_УКЦ_Группа_1_" in l:
      names_middle.append(l)
    if not ("Отв P'" in l):
      names_middle.append(l)
'''
for item in names_ecg:
  if item in names_middle:
    print(item)
    names_middle.remove(item)

for item in names_ecg:
  if item in names_min:
    print(item)
    names_min.remove(item)

def check_names(names):
  list1 = df.columns.tolist()
  list2 = df_test.columns.tolist()
  check_list = list(filter(lambda x: x in list1, list2))
  leng = 0
  cols = []

  for j in range(len(names)):
    for i in names[j]:
      if (i in check_list):
        cols.append(i)
        #print(i)
        leng+=1
  return cols

cols_min = check_names(names_min)
cols_middle = check_names(names_middle)
cols_max = check_names(names_max)

dfMin = pd.DataFrame()
dfMin = df.loc[:, cols_min]
dfMin_test = pd.DataFrame()
dfMin_test = df_test.loc[:, cols_min]

dfMiddle = pd.DataFrame()
dfMiddle = df.loc[:, cols_middle]
dfMiddle_test = pd.DataFrame()
dfMiddle_test = df_test.loc[:, cols_middle]

dfExtr = pd.DataFrame()
dfExtr = df.loc[:, cols_min]
dfExtr_test = pd.DataFrame()
dfExtr_test = df_test.loc[:, cols_min]

dfMax = pd.DataFrame()
dfMax = df.loc[:, cols_max]
dfMax_test = pd.DataFrame()
dfMax_test = df_test.loc[:, cols_max]



def accur(y_true, y_pred):
  accuracy = accuracy_score(y_true, y_pred) # меткость
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  fpr = fp / (fp + tn)
  fnr = fn / (fn + tp)
  specificity = tn/(fp + tn) # специфичность
  precision = precision_score(y_true, y_pred) # точность
  recall = recall_score(y_true, y_pred) # полнота/чувствительность

  f1 = f1_score(y_true, y_pred)
  nn = len(y_pred)
  n = tn+tp+fp+fn
  print("Accuracy           :", accuracy)
  print("F1-Score           :", f1)
  print("False Positive Rate:", fpr)
  print("False Negative Rate:", fnr)
  print("Precision          :", precision)
  print("Recall             :", recall)
  print("Specificity        :", specificity)
  print("tn                 :", tn,"{:.2f}%".format(100*tn/n))
  print("tp                 :", tp,"{:.2f}%".format(100*tp/n))
  print("fp (ошибка 1 рода) :", fp,"{:.2f}%".format(100*fp/n))
  print("fn (ошибка 2 рода) :", fn,"{:.2f}%".format(100*fn/n))

  return accuracy, tn, fp, fn, tp, precision, recall, f1, fpr, fnr, specificity

def f_model(X_train, Y_train, X_test, Y_test, str):
  scaler = StandardScaler()
  X_scaled_train = scaler.fit_transform(X_train)
  X_scaled_test = scaler.fit_transform(X_test)

  imputer = SimpleImputer(strategy='median')
  X_scaled_filled_train = imputer.fit_transform(X_scaled_train)
  X_scaled_filled_test = imputer.fit_transform(X_scaled_test)

  # X_train, X_test, Y_train, Y_test = train_test_split(X_scaled_filled, Y, test_size = 0.3, random_state=30, stratify=Y)

  linear_svc = svm.SVC(kernel='linear', C=1)
  random_forest = RandomForestClassifier()
  adaptive_boosting = AdaBoostClassifier()

  linear_svc.fit(X_scaled_filled_train, Y_train)
  random_forest.fit(X_scaled_filled_train, Y_train)
  adaptive_boosting.fit(X_scaled_filled_train, Y_train)

  # Оценка точности моделей
  results = {}
  models = ['Linear SVC', 'Random Forest', 'Voting Classifier', 'Adaptive Boosting']
  for model in models:
      if model == 'Linear SVC':
          y_pred = linear_svc.predict(X_scaled_filled_test)
      elif model == 'Random Forest':
          y_pred = random_forest.predict(X_scaled_filled_test)
      elif model == 'Adaptive Boosting':
          y_pred = adaptive_boosting.predict(X_scaled_filled_test)

      print('='*10, model, '='*10)
      accuracy, tn, fp, fn, tp, precision, recall, f1, fpr, fnr, specificity = accur(Y_test, y_pred)
      results[model] = [accuracy, tn, fp, fn, tp, precision, recall, f1, fpr, fnr, specificity]

  df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy', 'TN', 'FP', 'FN', 'TP', 'Precision', 'Recall', 'F1', 'FPR', 'FNR', 'Specificity'])
  print(str, '\n', df_results)

  # Сохранение таблицы с результатами в файл
  #df_results.to_csv('model_results.csv')

  return df_results

tableMax = f_model(dfMax.drop('label', axis=1), dfMax['label'], dfMax_test.drop('label', axis=1), dfMax_test['label'], "большое количество параметров")
tableMiddle = f_model(dfMiddle.drop('label', axis=1), dfMiddle['label'], dfMiddle_test.drop('label', axis=1), dfMiddle_test['label'], "среднее количество параметров")
#tableExtr = f_model(dfExtr.drop('label', axis=1), dfExtr['label'], dfExtr_test.drop('label', axis=1), dfExtr_test['label'], "минимальное количество параметров + экстремумы")
tableMin = f_model(dfMin.drop('label', axis=1), dfMin['label'], dfMin_test.drop('label', axis=1), dfMin_test['label'], "минимальное количество параметров")

with open('models_results.csv', 'w') as f:
  f.write('MAX\n')
  tableMax.to_csv(f, mode='a')
  f.write('\nMIDDLE\n')
  tableMiddle.to_csv(f, mode='a')
  f.write('\nMIN\n')
  tableMin.to_csv(f, mode='a')

def f_model(X_train, Y_train, X_test, Y_test, str):
  print('!'*15,str,'!'*15)
  scaler = StandardScaler()
  X_scaled_train = scaler.fit_transform(X_train)
  X_scaled_test = scaler.fit_transform(X_test)

  imputer = SimpleImputer(strategy='median')
  X_scaled_filled_train = imputer.fit_transform(X_scaled_train)
  X_scaled_filled_test = imputer.fit_transform(X_scaled_test)

  # X_train, X_test, Y_train, Y_test = train_test_split(X_scaled_filled, Y, test_size = 0.3, random_state=30, stratify=Y)

  linear_svc = svm.SVC(kernel='linear', C=1)
  random_forest = RandomForestClassifier()
  voting_classifier = VotingClassifier(estimators=[('rf', random_forest), ('svc', SVC()), ('abc', AdaBoostClassifier())])
  adaptive_boosting = AdaBoostClassifier()

  linear_svc.fit(X_scaled_filled_train, Y_train)
  random_forest.fit(X_scaled_filled_train, Y_train)
  voting_classifier.fit(X_scaled_filled_train, Y_train)
  adaptive_boosting.fit(X_scaled_filled_train, Y_train)

  # Оценка точности моделей
  results = {}
  feature_importances = {}
  models = ['Linear SVC', 'Random Forest', 'Voting Classifier', 'Adaptive Boosting']
  for model in models:
      if model == 'Linear SVC':
          y_pred = linear_svc.predict(X_scaled_filled_test)
          feature_importances[model] = linear_svc.coef_
      elif model == 'Random Forest':
          y_pred = random_forest.predict(X_scaled_filled_test)
          feature_importances[model] = random_forest.feature_importances_
      elif model == 'Voting Classifier':
          y_pred = voting_classifier.predict(X_scaled_filled_test)
          #feature_importances[model] = random_forest.feature_importances_
      elif model == 'Adaptive Boosting':
          y_pred = adaptive_boosting.predict(X_scaled_filled_test)
          feature_importances[model] = adaptive_boosting.feature_importances_

      print('='*10, model, '='*10)
      accuracy, tn, fp, fn, tp, precision, recall, f1, fpr, fnr, specificity = accur(Y_test, y_pred)
      results[model] = [accuracy, tn, fp, fn, tp, precision, recall, f1, fpr, fnr, specificity]

  df_results = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy', 'TN', 'FP', 'FN', 'TP', 'Precision', 'Recall', 'F1', 'FPR', 'FNR', 'Specificity'])
  #print(str, '\n', df_results)

  #df_feature_importances = pd.DataFrame.from_dict(feature_importances, orient='index')
  #print(str, '\n', df_feature_importances)
  #print(str, '\n', feature_importances)
  print('='*10, 'feature_importances', '='*10)
  for model in models:
    if (model != 'Voting Classifier'):
      print('='*5, model, '='*5)
      print(feature_importances[model])

  # Сохранение таблицы с результатами в файл
  #df_results.to_csv('model_results.csv')
  #df_feature_importances.to_csv('feature_importances.csv')

  return df_results, feature_importances

tableMax_res, tableMax_feat = f_model(dfMax.drop('label', axis=1), dfMax['label'], dfMax_test.drop('label', axis=1), dfMax_test['label'], "большое количество параметров")
tableMiddle_res, tableMiddle_feat = f_model(dfMiddle.drop('label', axis=1), dfMiddle['label'], dfMiddle_test.drop('label', axis=1), dfMiddle_test['label'], "среднее количество параметров")
#tableExtr_res, tableExtr_feat = f_model(dfExtr.drop('label', axis=1), dfExtr['label'], dfExtr_test.drop('label', axis=1), dfExtr_test['label'], "минимальное количество параметров + экстремумы")
tableMin_res, tableMin_feat = f_model(dfMin.drop('label', axis=1), dfMin['label'], dfMin_test.drop('label', axis=1), dfMin_test['label'], "минимальное количество параметров")

with open('models_results.csv', 'w') as f:
  f.write('MAX\n')
  tableMax_res.to_csv(f, mode='a')
  #f.write('\nMAX\n')
  #tableMax_feat.to_csv(f, mode='a')
  f.write('\nMIDDLE\n')
  tableMiddle_res.to_csv(f, mode='a')
  #f.write('\nMIDDLE\n')
  #tableMiddle_feat.to_csv(f, mode='a')
  f.write('\nMIN\n')
  tableMin_res.to_csv(f, mode='a')


def f1_score1(y_true, y_pred):
  classification_report(y_true, y_pred)
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  recall = true_positives / (possible_positives + K.epsilon())
  f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
  return f1


def f1_score2(y_true, y_pred):
    #y_true_np = tf.keras.backend.eval(y_true)
    #y_pred_np = tf.keras.backend.eval(y_pred)

    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    '''with tf.compat.v1.Session() as sess:
      y_true_np = y_true.numpy()
      y_pred_np = y_pred.numpy()
    print(y_true_np)
    print(y_pred_np)
    '''
    print(1)
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    support = precision_recall_fscore_support(y_true_np, y_pred_np, average='binary')
    print("Precision: ", support[0])
    print("Recall: ", support[1])
    print("F1 Score: ", support[2])
    print("Support: ", support[3])
    print(2)
    return f1

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    precision = tf.where(tf.math.is_nan(precision), tf.zeros_like(precision), precision)
    recall = tf.where(tf.math.is_nan(recall), tf.zeros_like(recall), recall)
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    print('true_positives: ', true_positives)
    return f1


from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd

# Открытие файла для записи результатов
file_result = open("result.txt", "w")

def f_model(X_train, Y_train, X_test, Y_test, method_name):
    # Масштабирование данных
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)

    # Заполнение отсутствующих значений
    imputer = SimpleImputer(strategy='median')
    X_scaled_filled_train = imputer.fit_transform(X_scaled_train)
    X_scaled_filled_test = imputer.transform(X_scaled_test)

    # Создание моделей
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    ada_clf = AdaBoostClassifier(n_estimators=50, learning_rate=1)


    # Обучение моделей
    rf_clf.fit(X_scaled_filled_train, Y_train)
    ada_clf.fit(X_scaled_filled_train, Y_train)

    # Предсказание результатов
    rf_predictions = rf_clf.predict(X_scaled_filled_test)
    ada_predictions = ada_clf.predict(X_scaled_filled_test)

    # Расчет метрик
    rf_accuracy = accuracy_score(Y_test, rf_predictions)
    ada_accuracy = accuracy_score(Y_test, ada_predictions)

    rf_confusion = confusion_matrix(Y_test, rf_predictions)
    ada_confusion = confusion_matrix(Y_test, ada_predictions)

    rf_precision = precision_score(Y_test, rf_predictions)
    ada_precision = precision_score(Y_test, ada_predictions)

    rf_recall = recall_score(Y_test, rf_predictions)
    ada_recall = recall_score(Y_test, ada_predictions)

    rf_f1 = f1_score(Y_test, rf_predictions)
    ada_f1 = f1_score(Y_test, ada_predictions)

    print(method_name, "Results:", file=file_result)
    print('-' * len(method_name), file=file_result)
    print(method_name, "Results:")
    print('-' * len(method_name))

    print("Random Forest Accuracy:", rf_accuracy, file=file_result)
    print("AdaBoost Accuracy:", ada_accuracy, file=file_result)

    print("Random Forest Confusion Matrix:", rf_confusion, file=file_result)
    print("AdaBoost Confusion Matrix:", ada_confusion, file=file_result)

    print("Random Forest Precision:", rf_precision, file=file_result)
    print("AdaBoost Precision:", ada_precision, file=file_result)

    print("Random Forest Recall:", rf_recall, file=file_result)
    print("AdaBoost Recall:", ada_recall, file=file_result)

    print("Random Forest F1 Score:", rf_f1, file=file_result)
    print("AdaBoost F1 Score:", ada_f1, file=file_result)
    print('=' * 45, file=file_result)
    print('=' * 45)

# Вызов функции для каждого случая
f_model(dfMax.drop('label', axis=1), dfMax['label'], dfMax_test.drop('label', axis=1), dfMax_test['label'], "большое количество параметров")
f_model(dfMiddle.drop('label', axis=1), dfMiddle['label'], dfMiddle_test.drop('label', axis=1), dfMiddle_test['label'], "среднее количество параметров")
f_model(dfExtr.drop('label', axis=1), dfExtr['label'], dfExtr_test.drop('label', axis=1), dfExtr_test['label'], "минимальное количество параметров + экстремумы")
f_model(dfMin.drop('label', axis=1), dfMin['label'], dfMin_test.drop('label', axis=1), dfMin_test['label'], "минимальное количество параметров")

file_result.close()