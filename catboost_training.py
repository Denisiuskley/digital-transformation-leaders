try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    get_ipython().run_line_magic('matplotlib', 'qt')
except:
    pass

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from datetime import datetime

from catboost import CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from umap import UMAP
import gc
import matplotlib.pyplot as plt
plt.close('all')
import seaborn as sns
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
scaler = StandardScaler()
'''
This script combine all results file and fitting boosting models
For embeddings used zise reduction technique
'''
def calc_metric(y_true, y_pred):
    auc = []
    for i, target in enumerate(['target_1', 'target_2', 'target_3', 'target_4']):
        auc.append(roc_auc_score(y_true[target], y_pred[target]))
    return np.mean(auc), np.std(auc)

def compress(train, val, test, n_comp, prefix, use_embs = False, nan_mode = ''):
    train = train.copy()
    val = val.copy()
    test = test.copy()
    alg_dict = {
        'tsvd': TruncatedSVD(n_components=n_comp, algorithm='arpack', random_state=42),
        'pca': PCA(n_components=n_comp, random_state=42),
     #    'umap': UMAP(n_components=n_comp, init='spectral', learning_rate=1.0,
     # local_connectivity=1.0, low_memory=True, n_neighbors=10, random_state=42, n_jobs=-1)
    }
    new_train = {}
    new_test = {}
    new_val = {}
    if use_embs:
        for col in train.columns:
            new_train[col] = list(train[col])
            new_val[col] = list(val[col])
            new_test[col] = list(test[col])
    else:
        col = 'client_id'
        new_train[col] = list(train[col])
        new_val[col] = list(val[col])
        new_test[col] = list(test[col])
    
    columns = list(train.columns)
    columns.remove('client_id')
    if nan_mode == 'mean':        
        for col in tqdm(train.columns, desc = 'fillna mean'):
            mean = np.nanmean(train[col])
            train.fillna(mean, inplace = True)
            val.fillna(mean, inplace = True)
            test.fillna(mean, inplace = True)
    elif nan_mode == 'zero':
        print('***fillna zero***')
        start = datetime.now()
        train.fillna(0, inplace = True)
        val.fillna(0, inplace = True)
        test.fillna(0, inplace = True)
        print(datetime.now() - start)
        
    for name, alg in alg_dict.items():
        print(f'*****{name}******')
        start = datetime.now()                
        train_alg = alg.fit_transform(scaler.fit_transform(train.drop(['client_id'], axis = 1))).reshape(-1, n_comp)
        val_alg = alg.transform(scaler.transform(val.drop(['client_id'], axis = 1))).reshape(-1, n_comp)
        test_alg = alg.transform(scaler.transform(test.drop(['client_id'], axis = 1))).reshape(-1, n_comp)                
        for i in range(n_comp):
            new_train[f'{prefix}_{name}_{i}'] = train_alg[:, i]
            new_val[f'{prefix}_{name}_{i}'] = val_alg[:, i]
            new_test[f'{prefix}_{name}_{i}'] = test_alg[:, i]
        print(datetime.now() - start)
    
    new_train = pd.DataFrame(new_train)
    new_test = pd.DataFrame(new_test)
    new_val = pd.DataFrame(new_val) 
    return new_train, new_val, new_test

def kmeans(train, val, test, n_comp, prefix, useOHE = False):
    kmeans = KMeans(n_clusters=n_comp, random_state=0, n_init="auto")
    name = 'kmeans'
    new_train = {}
    new_test = {}
    new_val = {}
    col = 'client_id'
    new_train[col] = list(train[col])
    new_val[col] = list(val[col])
    new_test[col] = list(test[col])
    
    train_alg = kmeans.fit_predict(train.drop(['client_id'], axis = 1))
    val_alg = kmeans.predict(val.drop(['client_id'], axis = 1))
    test_alg = kmeans.predict(test.drop(['client_id'], axis = 1))
    if useOHE:
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(train_alg.reshape(-1, 1))
        train_alg = enc.transform(train_alg.reshape(-1, 1)).toarray()
        val_alg = enc.transform(val_alg.reshape(-1, 1)).toarray()
        test_alg = enc.transform(test_alg.reshape(-1, 1)).toarray()               
        for i in range(n_comp):
            new_train[f'{prefix}_{name}_{i}'] = train_alg[:, i]
            new_val[f'{prefix}_{name}_{i}'] = val_alg[:, i]
            new_test[f'{prefix}_{name}_{i}'] = test_alg[:, i]
    else:
        new_train[f'{prefix}_{name}'] = train_alg
        new_val[f'{prefix}_{name}'] = val_alg
        new_test[f'{prefix}_{name}'] = test_alg
    new_train = pd.DataFrame(new_train)
    new_test = pd.DataFrame(new_test)
    new_val = pd.DataFrame(new_val)
    return new_train, new_val, new_test

start = datetime.now()

target_features = ['target_1', 'target_2', 'target_3', 'target_4',]
drop_features = target_features + ['client_id']

isSubmit = True #Submit or model hiperparameters tuning
path = 'created_data/'

# Read history features and targets
train_df = pd.read_parquet(path + "train_history_mean.parquet")
validate_df = pd.read_parquet(path + "validate_history_mean.parquet")
test_df = pd.read_parquet(path + "test_history_mean.parquet")
print('History', train_df.shape, validate_df.shape, test_df.shape)

# Read lifestream embeddings of transactions
train_trx_df = pd.read_parquet(path + "train.parquet").drop(target_features, axis = 1)
validate_trx_df = pd.read_parquet(path + "validate.parquet").drop(target_features, axis = 1)
test_trx_df = pd.read_parquet(path + "not_only_trx.parquet")
print('Transactions', train_trx_df.shape, validate_trx_df.shape, test_trx_df.shape)
train_trx_df, validate_trx_df, test_trx_df = \
    compress(train_trx_df, validate_trx_df, test_trx_df, 5, 'emb', use_embs = True, nan_mode = 'zero')
print('Transactions compress', train_trx_df.shape, validate_trx_df.shape, test_trx_df.shape)
train_df = train_df.merge(train_trx_df, on="client_id", how="left")
validate_df = validate_df.merge(validate_trx_df, on="client_id", how="left")
test_df = test_df.merge(test_trx_df, on="client_id", how="left")
print('Merged df', train_df.shape, validate_df.shape, test_df.shape)

# Read event type transaction features
train_trx_df = pd.read_parquet(path + "train_trx_f.parquet")
validate_trx_df = pd.read_parquet(path + "val_trx_f.parquet")
test_trx_df = pd.read_parquet(path + "test_trx_f.parquet")
print('Transactions', train_trx_df.shape, validate_trx_df.shape, test_trx_df.shape)
train_df = train_df.merge(train_trx_df, on="client_id", how="left")
validate_df = validate_df.merge(validate_trx_df, on="client_id", how="left")
test_df = test_df.merge(test_trx_df, on="client_id", how="left")

# Read redused sparse pivot table of 'currency', "src_type11", "dst_type11" transactions features
train_trx_df = pd.read_parquet(path + "train_trx_f3.parquet")
validate_trx_df = pd.read_parquet(path + "val_trx_f3.parquet")
test_trx_df = pd.read_parquet(path + "test_trx_f3.parquet")
print('Transactions', train_trx_df.shape, validate_trx_df.shape, test_trx_df.shape)
train_df = train_df.merge(train_trx_df, on="client_id", how="left")
validate_df = validate_df.merge(validate_trx_df, on="client_id", how="left")
test_df = test_df.merge(test_trx_df, on="client_id", how="left")

del train_trx_df, validate_trx_df, test_trx_df
while gc.collect():
    gc.collect()
print('Merged df', train_df.shape, validate_df.shape, test_df.shape)

# Read 2 month average text embeddings
train_text_df = pd.read_parquet(path + "train_text.parquet")
validate_text_df = pd.read_parquet(path + "validate_text.parquet")
test_text_df = pd.read_parquet(path + "test_text.parquet")
print('Texts', train_text_df.shape, validate_text_df.shape, test_text_df.shape)
train_text_df, validate_text_df, test_text_df = \
    compress(train_text_df, validate_text_df, test_text_df, 10, 'text', use_embs = False, nan_mode = 'zero')
print('Texts compress', train_text_df.shape, validate_text_df.shape, test_text_df.shape)
train_df = train_df.merge(train_text_df, on="client_id", how="left")
validate_df = validate_df.merge(validate_text_df, on="client_id", how="left")
test_df = test_df.merge(test_text_df, on="client_id", how="left")
print('Merged df', train_df.shape, validate_df.shape, test_df.shape)

# Read 1 month average text embeddings
train_text_df = pd.read_parquet(path + "train_text_1month.parquet")
validate_text_df = pd.read_parquet(path + "validate_text_1month.parquet")
test_text_df = pd.read_parquet(path + "test_text_1month.parquet")
print('Texts', train_text_df.shape, validate_text_df.shape, test_text_df.shape)
train_text_df, validate_text_df, test_text_df = \
    compress(train_text_df, validate_text_df, test_text_df, 10, 'text', use_embs = False, nan_mode = 'zero')
print('Texts compress', train_text_df.shape, validate_text_df.shape, test_text_df.shape)
train_df = train_df.merge(train_text_df, on="client_id", how="left")
validate_df = validate_df.merge(validate_text_df, on="client_id", how="left")
test_df = test_df.merge(test_text_df, on="client_id", how="left")

del train_text_df, validate_text_df, test_text_df
while gc.collect():
    gc.collect()
print('Merged df', train_df.shape, validate_df.shape, test_df.shape)

# Read geohashs embeddings. We can use 2 geohach id type and 2 key of ways to build ngrams
geohash = 'geohash_4' #geohash_4 geohash_5
key_geohash = 'client' #hash client
postfix = '20K'
train_geo_df = pd.read_parquet(path + f"train_{geohash}_{key_geohash}_{postfix}.parquet")
validate_geo_df = pd.read_parquet(path + f"validate_{geohash}_{key_geohash}_{postfix}.parquet")
test_geo_df = pd.read_parquet(path + f"test_{geohash}_{key_geohash}_{postfix}.parquet")
print('Geohashs', train_geo_df.shape, validate_geo_df.shape, test_geo_df.shape)

train_geo_df, validate_geo_df, test_geo_df = \
    compress(train_geo_df, validate_geo_df, test_geo_df, 5, 'geo', use_embs = False)
print('Geohashs compress', train_geo_df.shape, validate_geo_df.shape, test_geo_df.shape)

train_df = train_df.merge(train_geo_df, on="client_id", how="left")
validate_df = validate_df.merge(validate_geo_df, on="client_id", how="left")
test_df = test_df.merge(test_geo_df, on="client_id", how="left")
print('Merged df', train_df.shape, validate_df.shape, test_df.shape)

# Read geohashs count by client
train_geo_df = pd.read_parquet(path + f"train_{geohash}_{key_geohash}_num_{postfix}.parquet")
validate_geo_df = pd.read_parquet(path + f"validate_{geohash}_{key_geohash}_num_{postfix}.parquet")
test_geo_df = pd.read_parquet(path + f"test_{geohash}_{key_geohash}_num_{postfix}.parquet")
train_df = train_df.merge(train_geo_df, on="client_id", how="left")
validate_df = validate_df.merge(validate_geo_df, on="client_id", how="left")
test_df = test_df.merge(test_geo_df, on="client_id", how="left")
print('Merged df', train_df.shape, validate_df.shape, test_df.shape)

del train_geo_df, validate_geo_df, test_geo_df
while gc.collect():
    gc.collect()

if isSubmit:
    # Model fit on n folds
    n_fold = 5
    seeds = [0, 42, 21, 69, 7575]
        
    df = pd.concat([train_df, validate_df])
    df.index = range(len(df))
    
    X = df.drop(columns=drop_features)
    X_test = test_df.drop(columns=["client_id"])
    
    params = {
        'target_1': {'bootstrap_type': 'Bernoulli',
                      'depth': 10,
                      'l2_leaf_reg': 0.128,
                      'learning_rate': 0.035,
                      'max_bin': 34,
                      'grow_policy': 'SymmetricTree',
                      'min_data_in_leaf': 21,
                      'leaf_estimation_iterations': 1,
                      'subsample': 0.503,
                      'iterations': 20000,
                      'verbose': False,
                      'eval_metric': 'AUC',
                      'task_type': 'GPU',
                      'early_stopping_rounds': 250,
                      'use_best_model': True,
                      'random_state': 42,
                      'nan_mode': 'Min',
                    },
        'target_2': {'bootstrap_type': 'Bernoulli',
                      'depth': 9,
                      'l2_leaf_reg': 37,
                      'learning_rate': 0.0325,
                      'max_bin': 18,
                      'grow_policy': 'Lossguide',
                      'min_data_in_leaf': 136,
                      'leaf_estimation_iterations': 1,
                      'subsample': 0.148,
                      'iterations': 20000,
                      'verbose': False,
                      'eval_metric': 'AUC',
                      'task_type': 'GPU',
                      'early_stopping_rounds': 250,
                      'use_best_model': True,
                      'random_state': 42,
                      'nan_mode': 'Min',
            
                    },
        'target_3': {'bootstrap_type': 'Bernoulli',
                      'depth': 5,
                      'l2_leaf_reg': 0.0096,
                      'learning_rate': 0.03,
                      'max_bin': 59,
                      'grow_policy': 'Lossguide',
                      'min_data_in_leaf': 268,
                      'leaf_estimation_iterations': 2,
                      'subsample': 0.716,
                      'iterations': 20000,
                      'verbose': False,
                      'eval_metric': 'AUC',
                      'task_type': 'GPU',
                      'early_stopping_rounds': 250,
                      'use_best_model': True,
                      'random_state': 42,
                      'nan_mode': 'Max',
            
                    },
        'target_4': {'bootstrap_type': 'Bernoulli',
                      'depth': 5,
                      'l2_leaf_reg': 43,
                      'learning_rate': 0.03,
                      'max_bin': 15,
                      'grow_policy': 'Lossguide',
                      'min_data_in_leaf': 81,
                      'leaf_estimation_iterations': 4,
                      'subsample': 0.67,
                      'iterations': 20000,
                      'verbose': False,
                      'eval_metric': 'AUC',
                      'task_type': 'GPU',
                      'early_stopping_rounds': 250,
                      'use_best_model': True,
                      'random_state': 42,
                      'nan_mode': 'Min',
                    }
    }
    submission = pd.DataFrame([])
    submission["client_id"] = test_df["client_id"]
    submission[target_features] = 0
    
    val = pd.DataFrame([])
    val["client_id"] = df["client_id"]
    val[target_features] = 0    
    
    for index, target_col in enumerate(target_features):
        print(f'*****Strat training target: {target_col}*****')
        start_target = datetime.now()
        y = df[target_col].values
        param = params[target_col]
        for seed in seeds:
            skf = StratifiedKFold(n_splits = n_fold, shuffle = True, random_state = seed)
            for i, (train_index, val_index) in enumerate(skf.split(X, y)):
                
                model = CatBoostClassifier(**param)
                
                X_train = X.iloc[train_index]
                y_train = y[train_index]
                
                X_val = X.iloc[val_index]
                y_val = y[val_index]
            
                train_pool = Pool(X_train, y_train)
                eval_pool = Pool(X_val, y_val)    
                model.fit(train_pool, eval_set=eval_pool)
                
                submission[target_col] += model.predict_proba(X_test)[:, 1]
                val.loc[val_index, target_col] += model.predict_proba(X_val)[:, 1]
                
                print(f'SEED {seed} / {seeds} | Fold {i + 1} / {n_fold}: {roc_auc_score(y_val, val.loc[val_index, target_col]): .5f}')
        print(f'*****Calculation time of {target_col}: {datetime.now() - start_target}*****')            
    
    submission[target_features] = submission[target_features] / n_fold / len(seeds)
    val[target_features] /= len(seeds)

    auc, aucs = calc_metric(df[target_features], val)
    print(auc, aucs, auc - aucs/2)
    print(f'Calculation time: {datetime.now() - start}')
    submission.to_csv("submission.csv")

else:
    import telebot
    bot = telebot.TeleBot('')
    import optuna
    from time import sleep
    optuna.logging.set_verbosity(optuna.logging.WARN)
    
    n_trials = 25        
    n_jobs = 1
    X_train = train_df.drop(columns=drop_features)
    X_val = validate_df.drop(columns=drop_features)
    n_features = X_train.shape[1]
    for num in range(20):
        for index, target_col in enumerate(target_features): 
            y_train = train_df[target_col].values
            y_val = validate_df[target_col].values
            
            param = {'bootstrap_type': 'Bernoulli',
                      'depth': 5,
                      'l2_leaf_reg': 1,
                      'learning_rate': 0.06,
                      'max_bin': 40,
                      'grow_policy': 'Lossguide',
                      'min_child_samples': 10,
                      'subsample': 0.75,
                      'iterations': 10000,
                      'verbose': 500,
                      'eval_metric': 'AUC',
                      'task_type': 'GPU',
                      'early_stopping_rounds': 200,
                      'use_best_model': True,
                      'random_state': 42,
                      'nan_mode': 'Max',
                      }
            
            model = CatBoostClassifier(**param)
                                               
            train_pool = Pool(X_train, y_train)
            eval_pool = Pool(X_val, y_val)    
            model.fit(train_pool, eval_set=eval_pool)        
            y_pred = model.predict_proba(X_val)[:, 1]        
            metric = roc_auc_score(y_val, y_pred)        
            metric = np.round(metric, 5)
            
            bot.send_message(312849799, f'n prod = {target_col} start metric = {metric}')
            
            def logging_callback(study, frozen_trial):
                previous_best_value = study.user_attrs.get("previous_best_value", None)
                if previous_best_value != study.best_value:
                    study.set_user_attr("previous_best_value", study.best_value)
                    params = {param: value for param, value in frozen_trial.params.items() if 'fw' not in param}
        
                    mess = "Trial {} value: {} parameters: {}. ".format(
                    frozen_trial.number,
                    frozen_trial.value,
                    params)
                    try:
                        bot.send_message(312849799, mess)
                    except: pass
                    print(mess)
                    
            def objective(trial):
                param = {
                        'iterations': 10000,
                        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", 'MVS']),
                        'depth': trial.suggest_int("depth", 2, 12),            
                        'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1e-3, 100, log=True),
                        'learning_rate': trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
                        'max_bin': trial.suggest_int('max_bin', 2, 400, log=True),
                        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 300),
                        'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 5),
                        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
                        'nan_mode': trial.suggest_categorical('nan_mode', ['Min', 'Max']),                        
                        # 'grow_policy': 'Lossguide',
                        'verbose': 500,
                        'random_state': 42,                
                        'early_stopping_rounds': 200,
                        'use_best_model': True,
                        'task_type': 'GPU',
                        'eval_metric': 'AUC',
                    }
                if param["bootstrap_type"] == "MVS":
                    param["bootstrap_type"] = "Bernoulli"
                    
                if param["bootstrap_type"] == "Bayesian":
                    param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
                elif param["bootstrap_type"] == "Bernoulli":
                    param["subsample"] = trial.suggest_float("subsample", 0.01, 1)
                
                feature_weights = {}
                for i in range(n_features):
                    feature_weights[f"feature_weight_{i}"] = trial.suggest_int(f"fw_{i}", 0, 2) / 2
                param['feature_weights'] = list(feature_weights.values())
                
                model = CatBoostClassifier(**param)
                                                   
                train_pool = Pool(X_train, y_train)
                eval_pool = Pool(X_val, y_val)    
                model.fit(train_pool, eval_set=eval_pool)
                
                y_pred = model.predict_proba(X_val)[:, 1]            
                metric = roc_auc_score(y_val, y_pred)            
                metric = np.round(metric, 5)
                
                try:
                    best = float(joblib.load(f'best_metric_{target_col}.joblib'))
                except: best = 0
                if metric > best:
                    joblib.dump(metric, f'best_metric_{target_col}.joblib') 
                    joblib.dump(param, f'param_{target_col}.joblib')
                    model.save_model(f'models/{target_col}')
                while gc.collect():
                    gc.collect()
                sleep(5)
                return metric
                 
            pruner = optuna.pruners.HyperbandPruner()
            sampler = optuna.samplers.TPESampler()
            
            name = f'storage_{target_col}'
            storage = optuna.storages.RDBStorage(url=f'sqlite:///databases/{name}.db', engine_kwargs={"connect_args": {"timeout": 100}})
            study = optuna.create_study(study_name=name, sampler = sampler, pruner=pruner, direction="maximize", storage=storage, load_if_exists = True)
            study.optimize(objective, n_trials=n_trials, callbacks=[logging_callback], n_jobs=n_jobs)




