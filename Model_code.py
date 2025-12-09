import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

best_params = {'alpha': 0.3, 'lambda' :1, 'colsample_bytree': 0.3, 'learning_rate': 0.3,  'max_depth': 1, 'n_estimators': 500 }

Data = pd.read_csv('/Data_LFP.csv', sep=',' , header= 0 )
LFP_cycle, LFP_var, LFP_Tem = np.array(Data['cycle']), np.array(Data['var']).reshape(-1,1), np.array(Data['Tem'])
var_log, LFP_cycle, LFP_Tem =np.log10(LFP_var).reshape(-1,1), np.log10(LFP_cycle), LFP_Tem.astype(int) 
scaler = StandardScaler().fit(var_log)
LFP_var = scaler.transform(var_log)

index = np.where(LFP_cycle > 3.3)
big_var, big_cycle, big_Tem = LFP_var[index], LFP_cycle[index], LFP_Tem[index]
LFP_var, LFP_Tem, LFP_cycle = np.delete(LFP_var, index), np.delete(LFP_Tem, index), np.delete(LFP_cycle,index)
big_train = np.hstack((big_var.reshape(-1, 1), big_Tem.reshape(-1, 1)))
LFP_train = np.hstack((LFP_var.reshape(-1, 1), LFP_Tem.reshape(-1, 1)))
 
trainLFP_indices = np.random.choice(len(LFP_train), size= 81, replace=False)
testLFP_indices = np.setdiff1d(np.arange(len(LFP_train)), trainLFP_indices)
train_LFP_var, train_LFP_final, train_LFP_cycle = LFP_var[trainLFP_indices].reshape(-1, 1), LFP_train[trainLFP_indices], LFP_cycle[trainLFP_indices]
test_LFP_var, test_LFP_final, test_LFP_cycle = LFP_var[testLFP_indices].reshape(-1, 1), LFP_train[testLFP_indices], LFP_cycle[testLFP_indices]

train_LFP_var, train_LFP_final, train_LFP_cycle  = np.concatenate((big_var, train_LFP_var)), np.concatenate((big_train, train_LFP_final)), np.concatenate((big_cycle, train_LFP_cycle))

#여기서부터 NCA
NCA = pd.read_csv('/Data_NCA.csv', sep=',' , header= 0 )

NCA25 = NCA.loc[NCA['Tem'] == 25, [NCA.columns[0],NCA.columns[1],NCA.columns[2]]]
NCA25_var, NCA25_cycle, NCA25_Tem = np.array(NCA25['var']).reshape(-1,1), np.array(NCA25['cycle']),  np.array(NCA25['Tem'])
var_log1, NCA25_cycle, NCA25_Tem  = np.log10(NCA25_var).reshape(-1,1), np.log10(NCA25_cycle), NCA25_Tem.astype(int)
NCA25_var = scaler.transform(var_log1)
NCA25_train = np.hstack((NCA25_var, NCA25_Tem.reshape(-1, 1)))

NCA45 = NCA.loc[NCA['Tem'] == 45, [NCA.columns[0],NCA.columns[1],NCA.columns[2]]]
NCA45_var, NCA45_cycle, NCA45_Tem = np.array(NCA45['var']).reshape(-1,1), np.array(NCA45['cycle']),  np.array(NCA45['Tem'])
var_log2, NCA45_cycle, NCA25_Tem  = np.log10(NCA45_var).reshape(-1,1), np.log10(NCA45_cycle), NCA45_Tem.astype(int)
NCA45_var = scaler.transform(var_log2)
NCA45_train = np.hstack((NCA45_var, NCA45_Tem.reshape(-1, 1)))

#여기서부터 NCM
NCM = pd.read_csv('/Data_NCM.csv', sep=',' , header= 0 )

NCM25 = NCM.loc[NCM['Tem'] == 25, [NCM.columns[0],NCM.columns[1],NCM.columns[2]]]
NCM25_var, NCM25_cycle, NCM25_Tem = np.array(NCM25['var']).reshape(-1,1), np.array(NCM25['cycle']),  np.array(NCM25['Tem'])
var_log3, NCM25_cycle, NCM25_Tem  = np.log10(NCM25_var).reshape(-1,1), np.log10(NCM25_cycle), NCM25_Tem.astype(int)
NCM25_var = scaler.transform(var_log3)
NCM25_train = np.hstack((NCM25_var, NCM25_Tem.reshape(-1, 1)))

NCM45 = NCM.loc[NCM['Tem'] == 45, [NCM .columns[0], NCM .columns[1], NCM .columns[2]]]
NCM45_var, NCM45_cycle, NCM45_Tem = np.array(NCM45['var']), np.array(NCM45['cycle']),  np.array(NCM45['Tem'])
var_log4, NCM45_cycle, NCM45_Tem  = np.log10(NCM45_var).reshape(-1,1), np.log10(NCM45_cycle), NCM45_Tem.astype(int)
NCM45_var = scaler.transform(var_log4)
NCM45_train = np.hstack((NCM45_var, NCM45_Tem.reshape(-1, 1)))


### NCA index 분할
trainNCA25_indices = np.random.choice(len(NCA25_train), size=15, replace=False)
testNCA25_indices = np.setdiff1d(np.arange(len(NCA25_train)), trainNCA25_indices )
trainNCA45_indices = np.random.choice(len(NCA45_train), size=18, replace=False)
testNCA45_indices = np.setdiff1d(np.arange(len(NCA45_train)), trainNCA45_indices)
train_NCA25_var, train_NCA25_final, train_NCA25_cycle = NCA25_var[trainNCA25_indices],NCA25_train[trainNCA25_indices], NCA25_cycle[trainNCA25_indices]
test_NCA25_var, test_NCA25_final, test_NCA25_cycle = NCA25_var[testNCA25_indices],NCA25_train[testNCA25_indices], NCA25_cycle[testNCA25_indices]
train_NCA45_var, train_NCA45_final, train_NCA45_cycle = NCA45_var[trainNCA45_indices],NCA45_train[trainNCA45_indices], NCA45_cycle[trainNCA45_indices]
test_NCA45_var, test_NCA45_final, test_NCA45_cycle = NCA45_var[testNCA45_indices],NCA45_train[testNCA45_indices], NCA45_cycle[testNCA45_indices]

#온도 무관하게 그냥 섞어버리기 

NCA_var_train =  np.concatenate((train_NCA25_var,train_NCA45_var)).reshape(-1, 1)
NCA_final_train =  np.concatenate((train_NCA25_final,train_NCA45_final))
NCA_cycle_train =  np.concatenate((train_NCA25_cycle, train_NCA45_cycle))
NCA_var_test =np.concatenate((test_NCA25_var, test_NCA45_var))
NCA_final_test=  np.concatenate((test_NCA25_final, test_NCA45_final))
NCA_cycle_test =np.concatenate((test_NCA25_cycle, test_NCA45_cycle ))

### NCM index 분할
trainNCM25_indices = np.random.choice(len(NCM25_train), size= 15, replace=False)
testNCM25_indices = np.setdiff1d(np.arange(len(NCM25_train)), trainNCM25_indices)
trainNCM45_indices = np.random.choice(len(NCM45_train), size= 20, replace=False)
testNCM45_indices = np.setdiff1d(np.arange(len(NCM45_train)), trainNCM45_indices)
train_NCM25_var, train_NCM25_final, train_NCM25_cycle = NCM25_var[trainNCM25_indices], NCM25_train[trainNCM25_indices], NCM25_cycle[trainNCM25_indices]
test_NCM25_var, test_NCM25_final, test_NCM25_cycle = NCM25_var[testNCM25_indices], NCM25_train[testNCM25_indices], NCM25_cycle[testNCM25_indices]
train_NCM45_var, train_NCM45_final, train_NCM45_cycle = NCM45_var[trainNCM45_indices], NCM45_train[trainNCM45_indices], NCM45_cycle[trainNCM45_indices]
test_NCM45_var, test_NCM45_final, test_NCM45_cycle = NCM45_var[testNCM45_indices], NCM45_train[testNCM45_indices], NCM45_cycle[testNCM45_indices]

#온도 무관하게 그냥 섞어버리기 

NCM_var_train =  np.concatenate((train_NCM25_var, train_NCM45_var))
NCM_final_train =  np.concatenate((train_NCM25_final, train_NCM45_final))
NCM_cycle_train =  np.concatenate((train_NCM25_cycle, train_NCM45_cycle))
NCM_var_test =np.concatenate((test_NCM25_var, test_NCM45_var))
NCM_final_test=  np.concatenate((test_NCM25_final, test_NCM45_final ))
NCM_cycle_test =np.concatenate((test_NCM25_cycle, test_NCM45_cycle))



kfold = KFold(n_splits=3, shuffle=True, random_state=10)
LFP_list = []
LFP_list1 = []
NCA_list = []
NCA_list1  = []
NCM_list= []
NCM_list1= []

for train_index, test_index in kfold.split(train_LFP_var):
    X_train, X_test = train_LFP_var[train_index].reshape(-1,1), train_LFP_var[test_index].reshape(-1,1)
    y_train, y_test =train_LFP_cycle[train_index], train_LFP_cycle[test_index]
    xgb_model = xgb.XGBRegressor(**best_params)
    xgb_model.fit(X_train, y_train)

    # 예측 및 RMSE 계산
    pred = xgb_model.predict(test_LFP_var)
    rmse = np.sqrt(mean_squared_error(10**test_LFP_cycle, 10**pred))
    mape = mean_absolute_percentage_error(10**test_LFP_cycle,10**pred) *100
    LFP_list.append(rmse)
    LFP_list1.append(mape)
    
    pred = xgb_model.predict(NCA_var_test)
    rmse = np.sqrt(mean_squared_error(10**NCA_cycle_test, 10**pred))
    mape = mean_absolute_percentage_error(10**NCA_cycle_test, 10**pred) *100
    NCA_list.append(rmse)
    NCA_list1.append(mape)
    
    pred = xgb_model.predict(NCM_var_test)
    rmse = np.sqrt(mean_squared_error(10**NCM_cycle_test, 10**pred))
    mape = mean_absolute_percentage_error(10**NCM_cycle_test, 10**pred) *100
    NCM_list.append(rmse)
    NCM_list1.append(mape)



# 모델 2: LFP와 NCA 데이터로 훈련
var_LFP_NCA = np.concatenate([train_LFP_var, train_NCA25_var ,train_NCA45_var]).reshape(-1, 1)
cycle_LFP_NCA = np.concatenate([train_LFP_cycle, train_NCA25_cycle,  train_NCA45_cycle])

NCALFP_list = []
NCALFP_list1 = []
NCANCA_list = []
NCANCA_list1 = []
NCANCM_list = []
NCANCM_list1 = []

for train_index, test_index in kfold.split(var_LFP_NCA):
    X_train, X_test = var_LFP_NCA[train_index].reshape(-1,1), var_LFP_NCA[test_index].reshape(-1,1)
    y_train, y_test = cycle_LFP_NCA[train_index].reshape(-1,1),cycle_LFP_NCA[test_index].reshape(-1,1)
    xgb_model = xgb.XGBRegressor(**best_params)
    xgb_model.fit(X_train, y_train)

    # 예측 및 RMSE 계산
    pred = xgb_model.predict(test_LFP_var)
    rmse = np.sqrt(mean_squared_error(10**test_LFP_cycle, 10**pred))
    mape = mean_absolute_percentage_error(10**test_LFP_cycle,10**pred) *100
    NCALFP_list.append(rmse)
    NCALFP_list1.append(mape)
    
    pred = xgb_model.predict(NCA_var_test)
    rmse = np.sqrt(mean_squared_error(10**NCA_cycle_test, 10**pred))
    mape = mean_absolute_percentage_error(10**NCA_cycle_test,10**pred) *100
    NCANCA_list.append(rmse)
    NCANCA_list1.append(mape)
    
    pred = xgb_model.predict(NCM_var_test)
    rmse = np.sqrt(mean_squared_error(10**NCM_cycle_test, 10**pred))
    mape = mean_absolute_percentage_error(10**NCM_cycle_test,10**pred) *100
    NCANCM_list.append(rmse)
    NCANCM_list1.append(mape)
    
    
    
# 모델 3: LFP, NCA, NCM 데이터로 훈련
var_LFP_NCA_NCM = np.concatenate([train_LFP_var, train_NCA25_var ,train_NCA45_var, train_NCM25_var ,train_NCM45_var]).reshape(-1, 1)
cycle_LFP_NCA_NCM = np.concatenate([train_LFP_cycle, train_NCA25_cycle, train_NCA45_cycle, train_NCM25_cycle ,train_NCM45_cycle])
Var2_train =  np.concatenate((train_LFP_final, train_NCA25_final, train_NCA45_final, train_NCM25_final, train_NCM45_final ))
Cycle2_train =np.concatenate((train_LFP_cycle,train_NCA25_cycle, train_NCA45_cycle, train_NCM25_cycle ,train_NCM45_cycle))

    
NCMLFP_list = []
NCMLFP_list1 = []
NCMNCA_list= []
NCMNCA_list1 = []
NCMNCM_list = []
NCMNCM_list1 = []

for train_index, test_index in kfold.split(var_LFP_NCA_NCM):
    X_train1, X_train2, X_test = var_LFP_NCA_NCM[train_index].reshape(-1,1), Var2_train[train_index], var_LFP_NCA_NCM[test_index].reshape(-1,1)
    y_train1, y_train2, y_test = cycle_LFP_NCA_NCM[train_index], Cycle2_train[train_index], cycle_LFP_NCA_NCM[test_index]
    xgb_model = xgb.XGBRegressor(**best_params)
    xgb_model.fit(X_train1, y_train1)

    # 예측 및 RMSE 계산
    pred = xgb_model.predict(test_LFP_var)
    rmse = np.sqrt(mean_squared_error(10**test_LFP_cycle, 10**pred))
    mape = mean_absolute_percentage_error(10**test_LFP_cycle,10**pred) *100
    NCMLFP_list.append(rmse)
    NCMLFP_list1.append(mape)
    
    pred = xgb_model.predict(NCA_var_test)
    rmse = np.sqrt(mean_squared_error(10**NCA_cycle_test, 10**pred))
    mape = mean_absolute_percentage_error(10**NCA_cycle_test,10**pred) *100
    NCMNCA_list.append(rmse)
    NCMNCA_list1.append(mape)
    
    pred = xgb_model.predict(NCM_var_test)
    rmse = np.sqrt(mean_squared_error(10**NCM_cycle_test, 10**pred))
    mape = mean_absolute_percentage_error(10**NCM_cycle_test, 10**pred) *100
    NCMNCM_list.append(rmse)
    NCMNCM_list1.append(mape)