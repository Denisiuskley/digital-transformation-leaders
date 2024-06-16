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
from ptls.preprocessing import PandasDataPreprocessor
from tqdm import tqdm

'''
Create history features and targets
History features: mean for all history and for previous month
'''

def prepare_embeddings(dict_rows):
    pandas_dict = {
        "client_id": [],
        "target_1": [],
        "target_2": [],
        "target_3": [],
        "target_4": [],
        "history_all_1": [],
        "history_all_2": [],
        "history_all_3": [],
        "history_all_4": [],
        "history_2_1": [],
        "history_2_2": [],
        "history_2_3": [],
        "history_2_4": [],
        "month": [],
    }
    for features in tqdm(dict_rows, total = len(dict_rows)):
        t1 = features["target_1"].numpy()
        t2 = features["target_2"].numpy()
        t3 = features["target_3"].numpy()
        t4 = features["target_4"].numpy()
        for month in range(1, 13):
            try:                
                from_2 =  max(0, month - 2)                
                from_all = 0
                to_ = month - 1
                tt1 = int(t1[to_])
                tt2 = int(t2[to_])
                tt3 = int(t3[to_])
                tt4 = int(t4[to_])                
               
                th12 = t1[from_2: to_]
                th22 = t2[from_2: to_]
                th32 = t3[from_2: to_]
                th42 = t4[from_2: to_]                
                
                ta1 = t1[from_all: to_]
                ta2 = t2[from_all: to_]
                ta3 = t3[from_all: to_]
                ta4 = t4[from_all: to_]
                
                pandas_dict["target_1"].append(tt1)
                pandas_dict["target_2"].append(tt2)
                pandas_dict["target_3"].append(tt3)
                pandas_dict["target_4"].append(tt4)

                pandas_dict["history_all_1"].append(0 if len(ta1) == 0 else ta1.mean())
                pandas_dict["history_all_2"].append(0 if len(ta2) == 0 else ta2.mean())
                pandas_dict["history_all_3"].append(0 if len(ta3) == 0 else ta3.mean())
                pandas_dict["history_all_4"].append(0 if len(ta4) == 0 else ta4.mean())
                
                pandas_dict["history_2_1"].append(0 if len(th12) == 0 else th12.mean())
                pandas_dict["history_2_2"].append(0 if len(th22) == 0 else th22.mean())
                pandas_dict["history_2_3"].append(0 if len(th32) == 0 else th32.mean())
                pandas_dict["history_2_4"].append(0 if len(th42) == 0 else th42.mean())
 
                pandas_dict["month"].append(month)
                pandas_dict["client_id"].append(features["client_id"] + '_month=' + str(month))
            except: pass              
            
    return pd.DataFrame.from_dict(pandas_dict)

def prepare_embeddings_test(dict_rows):
    pandas_dict = {
        "client_id": [],
        "history_all_1": [],
        "history_all_2": [],
        "history_all_3": [],
        "history_all_4": [],
        "history_2_1": [],
        "history_2_2": [],
        "history_2_3": [],
        "history_2_4": [],
        "month": [],
    }
    for features in tqdm(dict_rows, total = len(dict_rows)):
        month, indexes = np.unique(features["mon"], return_index=True)
        targets1 = features["target_1"].numpy()[indexes]
        targets2 = features["target_2"].numpy()[indexes]
        targets3 = features["target_3"].numpy()[indexes]
        targets4 = features["target_4"].numpy()[indexes]
        from_2 =  max(0, len(targets1) - 1)
        from_all = 0

        pandas_dict["history_2_1"].append(targets1[from_2:].mean())
        pandas_dict["history_2_2"].append(targets2[from_2:].mean())
        pandas_dict["history_2_3"].append(targets3[from_2:].mean())
        pandas_dict["history_2_4"].append(targets4[from_2:].mean())

        pandas_dict["history_all_1"].append(targets1[from_all:].mean())
        pandas_dict["history_all_2"].append(targets2[from_all:].mean())
        pandas_dict["history_all_3"].append(targets3[from_all:].mean())
        pandas_dict["history_all_4"].append(targets4[from_all:].mean())
        pandas_dict["client_id"].append(features["client_id"])
        
        pandas_dict["month"].append(len(month) + 1)
            
    return pd.DataFrame.from_dict(pandas_dict)

target_train = pd.read_parquet("Hackathon/train_target.parquet")
test_target_b = pd.read_parquet("Hackathon/test_target_b.parquet")

target_preprocessor = PandasDataPreprocessor(
    col_id="client_id",
    col_event_time="mon",
    event_time_transformation='none',
    cols_identity=["target_1", "target_2", "target_3", "target_4"],
    return_records=False,
)

# Use lifestream preprocessor for data preparing
processed_target = target_preprocessor.fit_transform(target_train)
processed_target_test = target_preprocessor.transform(test_target_b)

processed_target_dict = processed_target.to_dict("records")
processed_target_test_dict = processed_target_test.to_dict("records")

history_train_df = prepare_embeddings(processed_target_dict)
history_validate_df  = prepare_embeddings(processed_target_test_dict)

history_test_df = prepare_embeddings_test(processed_target_test_dict)

history_train_df.to_parquet("created_data/train_history_mean.parquet", index=False, engine="pyarrow", compression="snappy")
history_validate_df.to_parquet("created_data/validate_history_mean.parquet", index=False, engine="pyarrow", compression="snappy")
history_test_df.to_parquet("created_data/test_history_mean.parquet", index=False, engine="pyarrow", compression="snappy")

