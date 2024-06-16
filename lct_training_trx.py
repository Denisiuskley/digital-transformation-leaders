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
from datetime import datetime

torch.set_float32_matmul_precision('medium')

class GetSplit(IterableProcessingDataset):
    def __init__(
        self,
        start_month,
        end_month,
        year=2022,
        col_id='client_id',
        col_time='event_time'
    ):
        super().__init__()
        self.start_month = start_month
        self.end_month = end_month
        self._year = year
        self._col_id = col_id
        self._col_time = col_time

    def __iter__(self):
        for rec in self._src:
            for month in range(self.start_month, self.end_month+1):
                features = rec[0] if type(rec) is tuple else rec
                features = features.copy()

                if month == 12:
                    month_event_time = datetime(self._year + 1, 1, 1).timestamp()
                else:
                    month_event_time = datetime(self._year, month + 1, 1).timestamp()

                year_event_time = datetime(self._year, 1, 1).timestamp()

                mask = features[self._col_time] < month_event_time
                
                if len(features["target_1"]) <= month - 1: continue
                for key, tensor in features.items():
                    if key.startswith('target'):
                        features[key] = tensor[month - 1].tolist()
                    elif key != self._col_id:
                        features[key] = tensor[mask]

                features[self._col_id] += '_month=' + str(month)

                yield features

def collate_feature_dict_with_target(batch, col_id='client_id', targets=False):
    batch_ids = []
    target_cols = []
    for sample in batch:
        batch_ids.append(sample[col_id])
        del sample[col_id]

        if targets:
            target_cols.append([sample[f'target_{i}'] for i in range(1, 5)])
            del sample['target_1']
            del sample['target_2']
            del sample['target_3']
            del sample['target_4']

    padded_batch = collate_feature_dict(batch)
    if targets:
        return padded_batch, batch_ids, target_cols
    return padded_batch, batch_ids


class InferenceModuleMultimodal(pl.LightningModule):
    def __init__(self, model, pandas_output=True, drop_seq_features=True, model_out_name='out'):
        super().__init__()

        self.model = model
        self.pandas_output = pandas_output
        self.drop_seq_features = drop_seq_features
        self.model_out_name = model_out_name

    def forward(self, x):
        x_len = len(x)
        if x_len == 3:
            x, batch_ids, target_cols = x
        else:
            x, batch_ids = x

        out = self.model(x)
        if x_len == 3:
            target_cols = torch.tensor(target_cols)
            x_out = {
                'client_id': batch_ids,
                'target_1': target_cols[:, 0],
                'target_2': target_cols[:, 1],
                'target_3': target_cols[:, 2],
                'target_4': target_cols[:, 3],
                self.model_out_name: out
            }
        else:
            x_out = {
                'client_id': batch_ids,
                self.model_out_name: out
            }
        torch.cuda.empty_cache()

        if self.pandas_output:
            return self.to_pandas(x_out)
        return x_out

    @staticmethod
    def to_pandas(x):
        expand_cols = []
        scalar_features = {}

        for k, v in x.items():
            if type(v) is torch.Tensor:
                v = v.cpu().numpy()

            if type(v) is list or len(v.shape) == 1:
                scalar_features[k] = v
            elif len(v.shape) == 2:
                expand_cols.append(k)
            else:
                scalar_features[k] = None

        dataframes = [pd.DataFrame(scalar_features)]
        for col in expand_cols:
            v = x[col].cpu().numpy()
            dataframes.append(pd.DataFrame(v, columns=[f'{col}_{i:04d}' for i in range(v.shape[1])]))

        return pd.concat(dataframes, axis=1)
    

transactions_train = pd.read_parquet("Hackathon/trx_train.parquet")
transactions_test = pd.read_parquet("Hackathon/trx_test.parquet")

print('Dataset red')

preprocessor = PandasDataPreprocessor(
    col_id="client_id",
    col_event_time="event_time",
    event_time_transformation="dt_to_timestamp",
    cols_category=["event_type",
                   "event_subtype",
                   "currency",
                   "src_type11",
                   "src_type12",
                   "dst_type11",
                   "dst_type12",
                   "src_type21",
                   "src_type22",
                   "src_type31",
                   "src_type32"],
    cols_identity="amount",
    return_records=False,
)

processed_train = preprocessor.fit_transform(transactions_train)
processed_test = preprocessor.transform(transactions_test)

print('Processed done')

target_train = pd.read_parquet("Hackathon/train_target.parquet")
target_preprocessor = PandasDataPreprocessor(
    col_id="client_id",
    col_event_time="mon",
    event_time_transformation="dt_to_timestamp",
    cols_identity=["target_1", "target_2", "target_3", "target_4"],
    return_records=False,
)

processed_target = target_preprocessor.fit_transform(target_train)
test_target_b = pd.read_parquet("Hackathon/test_target_b.parquet")
processed_target_test = target_preprocessor.transform(test_target_b)

print('Processed target done')

train = MemoryMapDataset(
    data=processed_train.to_dict("records"),
    i_filters=[
        FeatureFilter(drop_feature_names=['client_id', 'target_1', 'target_2', 'target_3', 'target_4']),
        SeqLenFilter(min_seq_len=32),
        ISeqLenLimit(max_seq_len=4096),
        ToTorch()
    ]
)

test = MemoryMapDataset(
    data=processed_test.to_dict("records"),
    i_filters=[
        FeatureFilter(drop_feature_names=['client_id', 'target_1', 'target_2', 'target_3', 'target_4']),
        SeqLenFilter(min_seq_len=32),
        ISeqLenLimit(max_seq_len=4096),
        ToTorch()
    ]
)

train_ds = ColesIterableDataset(
    data=train,
    splitter=SampleSlices(
        split_count=5,
        cnt_min=32,
        cnt_max=360
    )
)

valid_ds = ColesIterableDataset(
    data=test,
    splitter=SampleSlices(
        split_count=5,
        cnt_min=32,
        cnt_max=360
    )
)

train_dl = PtlsDataModule(
    train_data=train_ds,
    train_num_workers=0,
    train_batch_size=256,
    valid_data=valid_ds,
    valid_num_workers=0,
    valid_batch_size=256
)

trx_out = 24
trx_encoder_params = dict(
    embeddings_noise=0.003,
    numeric_values={'amount': 'log'},
    embeddings={
        "event_type": {'in': preprocessor.get_category_dictionary_sizes()["event_type"], "out": trx_out},
        "event_subtype": {'in': preprocessor.get_category_dictionary_sizes()["event_subtype"], "out": trx_out},
        'src_type11': {'in': preprocessor.get_category_dictionary_sizes()["src_type11"], 'out': trx_out},
        'src_type12': {'in': preprocessor.get_category_dictionary_sizes()["src_type12"], 'out': trx_out},
        'dst_type11': {'in': preprocessor.get_category_dictionary_sizes()["dst_type11"], 'out': trx_out},
        'dst_type12': {'in': preprocessor.get_category_dictionary_sizes()["dst_type12"], 'out': trx_out},
        'src_type22': {'in': preprocessor.get_category_dictionary_sizes()["src_type22"], 'out': trx_out},
        'src_type31': {'in': preprocessor.get_category_dictionary_sizes()["src_type31"], 'out': trx_out},
        'src_type32': {'in': preprocessor.get_category_dictionary_sizes()["src_type32"], 'out': trx_out},
      }
)

seq_encoder = RnnSeqEncoder(
    trx_encoder=TrxEncoder(**trx_encoder_params),
    hidden_size=256,
    type='gru',
)

model = CoLESModule(
    seq_encoder=seq_encoder,
    optimizer_partial=partial(torch.optim.Adam, lr=0.002),
    lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=3, gamma=0.9025)
)

trainer = pl.Trainer(
    max_epochs=100,
    limit_val_batches=5000,
    enable_progress_bar=True,
    gradient_clip_val=0.5,
    accumulate_grad_batches=5,
    logger=pl.loggers.TensorBoardLogger(
        save_dir='./logdir',
        name='baseline_result'
    ),
    callbacks=[
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
        pl.callbacks.ModelCheckpoint(every_n_train_steps=5000, save_top_k=-1),
    ]
)
print('Model train')
trainer.fit(model, train_dl)
torch.save(model.state_dict(), './model.pt')
model.load_state_dict(torch.load('./model.pt'))
print('Model train done')



validate = MemoryMapDataset(
    data=processed_test.merge(processed_target_test.drop("event_time", axis=1), on="client_id", how="inner").to_dict("records"),
    i_filters=[
        ISeqLenLimit(max_seq_len=4096),
        FeatureFilter(keep_feature_names=['client_id', 'target_1', 'target_2', 'target_3', 'target_4']),
        GetSplit(start_month=1, end_month=11),
        ToTorch(),
    ]
)

inference_validate_dl = DataLoader(
        dataset=validate,
        collate_fn=partial(collate_feature_dict_with_target, targets=True),
        shuffle=False,
        num_workers=0,
        batch_size=256,
    )

train = MemoryMapDataset(
    data=processed_train.merge(processed_target.drop("event_time", axis=1), on="client_id", how="inner").to_dict("records"),
    i_filters=[
        ISeqLenLimit(max_seq_len=4096),
        FeatureFilter(keep_feature_names=['client_id', 'target_1', 'target_2', 'target_3', 'target_4']),
        GetSplit(start_month=1, end_month=12),
        ToTorch(),
    ]
)


test = MemoryMapDataset(
    data=processed_test.to_dict("records"),
    i_filters=[
        ISeqLenLimit(max_seq_len=4096),
        FeatureFilter(keep_feature_names=['client_id', 'target_1', 'target_2', 'target_3', 'target_4']),
        ToTorch(),
    ]
)

inference_train_dl = DataLoader(
        dataset=train,
        collate_fn=partial(collate_feature_dict_with_target, targets=True),
        shuffle=False,
        num_workers=0,
        batch_size=256,
    )

inference_test_dl = DataLoader(
        dataset=test,
        collate_fn=collate_feature_dict_with_target,
        shuffle=False,
        num_workers=0,
        batch_size=256,
    )


inf_module = InferenceModuleMultimodal(
        model=model,
        pandas_output=True,
        drop_seq_features=True,
        model_out_name='emb',
    )

trainer = pl.Trainer(max_epochs=-1)

inf_validate_embeddings = pd.concat(
        trainer.predict(inf_module, inference_validate_dl)
    )
inf_validate_embeddings.to_parquet("created_data/validate_100.parquet", index=False, engine="pyarrow", compression="snappy")

inf_test_embeddings = pd.concat(
        trainer.predict(inf_module, inference_test_dl)
    )
inf_test_embeddings.to_parquet("created_data/test_100.parquet", index=False, engine="pyarrow", compression="snappy")



inf_train_embeddings = pd.concat(
        trainer.predict(inf_module, inference_train_dl)
    )

inf_train_embeddings.to_parquet("created_data/train_100.parquet", index=False, engine="pyarrow", compression="snappy")


not_only_trx = pd.DataFrame({"client_id": test_target_b["client_id"].unique()}).merge(inf_test_embeddings, how="left")

not_only_trx.to_parquet("created_data/not_only_trx_100.parquet", index=False, engine="pyarrow", compression="snappy")
    



