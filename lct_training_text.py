try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
    get_ipython().run_line_magic('matplotlib', 'qt')
except:
    pass

import pandas as pd
import numpy as np
import torch
from functools import partial
import pytorch_lightning as pl
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader

from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing.iterable_seq_len_limit import ISeqLenLimit
from ptls.data_load.iterable_processing.to_torch_tensor import ToTorch
from ptls.data_load.iterable_processing.feature_filter import FeatureFilter
from ptls.nn import TrxEncoder, RnnSeqEncoder
from ptls.frames.coles import CoLESModule
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesIterableDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.frames import PtlsDataModule
from ptls.preprocessing import PandasDataPreprocessor
from ptls.data_load.utils import collate_feature_dict
from ptls.data_load.iterable_processing_dataset import IterableProcessingDataset

from tqdm import tqdm
from datetime import datetime

def prepare_embeddings(dict_rows):
    pandas_dict = {
        "client_id": [],
        "embedding": []
    }
    for features in dict_rows:
         for month in range(1, 13):
            features = features.copy()
            
            if month == 12:
                month_event_time = datetime(2022 + 1, 1, 1).timestamp()
            else:
                month_event_time = datetime(2022, month + 1, 1).timestamp()
                
            if month == 1:
                prev_month_event_time = datetime(2021, 12, 1).timestamp()
            else:
                prev_month_event_time = datetime(2022, month - 1, 1).timestamp()
         
            mask = (features["event_time"] < month_event_time) &  (features["event_time"] >= prev_month_event_time)
            mask = mask.numpy()
            embeddings = np.stack(features["embedding"], axis=0)

            emb_slice = embeddings[mask]
            if len(emb_slice) > 0:
                pandas_dict["embedding"].append(emb_slice.mean(axis=0))
                pandas_dict["client_id"].append(features["client_id"] + '_month=' + str(month))
            
    return pd.DataFrame.from_dict(pandas_dict)

def prepare_embeddings_test(dict_rows):
    pandas_dict = {
        "client_id": [],
        "embedding": []
    }
    for features in dict_rows:
        month = len(np.unique(features["mon"]).tolist()) + 1
        features = features.copy()
        #print(month)
        if month == 12:
            month_event_time = datetime(2022 + 1, 1, 1).timestamp()
        elif month == 13:
            month_event_time = datetime(2022 + 1, 2, 1).timestamp()
        else:
            month_event_time = datetime(2022, month + 1, 1).timestamp()
                
        if month == 1:
            prev_month_event_time = datetime(2021, 12, 1).timestamp()
        else:
            prev_month_event_time = datetime(2022, month - 1, 1).timestamp()
         
        mask = (features["event_time"] < month_event_time) & (features["event_time"] >= prev_month_event_time)
        mask = mask.numpy()
        embeddings = np.stack(features["embedding"], axis=0)

        emb_slice = embeddings[mask]
        if len(emb_slice) > 0:
            pandas_dict["embedding"].append(emb_slice.mean(axis=0))
            pandas_dict["client_id"].append(features["client_id"])
            
    return pd.DataFrame.from_dict(pandas_dict)

dial_train = pd.read_parquet("Hackathon/dial_train.parquet")
dial_test = pd.read_parquet("Hackathon/dial_test.parquet")
test_target_b = pd.read_parquet("Hackathon/test_target_b.parquet")
target_train = pd.read_parquet("Hackathon/train_target.parquet")

preprocessor = PandasDataPreprocessor(
    col_id="client_id",
    col_event_time="event_time",
    event_time_transformation="dt_to_timestamp",
    cols_identity="embedding",
    return_records=False,
)

target_preprocessor = PandasDataPreprocessor(
    col_id="client_id",
    col_event_time="mon",
    event_time_transformation='none',
    cols_identity=["target_1", "target_2", "target_3", "target_4"],
    return_records=False,
)

processed_train = preprocessor.fit_transform(dial_train)
processed_test = preprocessor.transform(dial_test)
processed_target = target_preprocessor.fit_transform(target_train)
processed_target_test = target_preprocessor.transform(test_target_b)

merged_validate = processed_test.merge(
    processed_target_test.drop(["event_time"], axis=1), on="client_id", how="inner").to_dict("records")
merged_train = processed_train.merge(
    processed_target.drop(["event_time"], axis=1), on="client_id", how="inner").to_dict("records")
df_train_result = prepare_embeddings(merged_train)
df_validate_result = prepare_embeddings(merged_validate)

df_test_result = prepare_embeddings_test(merged_validate)

cols = [f"text_{i}"  for i in range(768)]
df_train_result[cols] = pd.DataFrame(df_train_result["embedding"].tolist(), index=df_train_result.index)
df_validate_result[cols] = pd.DataFrame(df_validate_result["embedding"].tolist(), index=df_validate_result.index)
df_test_result[cols] = pd.DataFrame(df_test_result["embedding"].tolist(), index=df_test_result.index)

df_train_result.drop("embedding", axis=1, inplace=True)
df_validate_result.drop("embedding", axis=1, inplace=True)
df_test_result.drop("embedding", axis=1, inplace=True)

df_train_result.to_parquet("created_data/train_text.parquet", index=False, engine="pyarrow", compression="snappy")
df_validate_result.to_parquet("created_data/validate_text.parquet", index=False, engine="pyarrow", compression="snappy")
df_test_result.to_parquet("created_data/test_text.parquet", index=False, engine="pyarrow", compression="snappy")



