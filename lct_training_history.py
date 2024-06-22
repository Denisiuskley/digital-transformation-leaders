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
        "month": [],
        "len": [],
    }
    for i in range(1, 12):
        for j in range(1, 5):
            pandas_dict[f"history_{j}_{i}"] = []
    
    for features in tqdm(dict_rows, total = len(dict_rows)):
        t = [features["target_1"].numpy(),
        features["target_2"].numpy(),
        features["target_3"].numpy(),
        features["target_4"].numpy()]
        for month in range(9, 13):
            try: 
                to_ = month - 1
                for j in range(1, 5):
                    target = t[j - 1]
                    pandas_dict[f"target_{j}"].append(int(target[to_]))
                    for i in range(1, 12):
                        from_ =  max(0, month - i - 1)
                        th = target[from_: to_]
                        if len(th) == 0:
                            pandas_dict[f"history_{j}_{i}"].append(0)
                        else:
                            pandas_dict[f"history_{j}_{i}"].append(th.mean())
                        
                pandas_dict["month"].append(month)
                pandas_dict["len"].append(len(th))
                pandas_dict["client_id"].append(features["client_id"] + '_month=' + str(month))
            except: pass              
            
    return pd.DataFrame.from_dict(pandas_dict)

def prepare_embeddings_test(dict_rows):
    pandas_dict = {
        "client_id": [],
        "month": [],
        "len": [],
    }
    for i in range(1, 12):
        for j in range(1, 5):
            pandas_dict[f"history_{j}_{i}"] = []
            
    for features in tqdm(dict_rows, total = len(dict_rows)):
        month, indexes = np.unique(features["mon"], return_index=True)
        
        t = [features["target_1"].numpy()[indexes],
        features["target_2"].numpy()[indexes],
        features["target_3"].numpy()[indexes],
        features["target_4"].numpy()[indexes]]
        lt = len(t[0])
        for j in range(1, 5):
            target = t[j - 1]
            for i in range(1, 12):
                from_ =  max(0, lt - i)
                th = target[from_:]
                pandas_dict[f"history_{j}_{i}"].append(th.mean())

        pandas_dict["client_id"].append(features["client_id"])       
        pandas_dict["month"].append(len(month) + 1)
        pandas_dict["len"].append(lt)    
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
