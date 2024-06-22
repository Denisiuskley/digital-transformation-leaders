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

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium')

'''
Greate geofeatures. Main idea is to create embedding vectors for every clients.
Embeddings characterize the conditional distance between clients by analogy with ngrams.
Triplet loss was used.
'''
		
class GeoHashDataset(Dataset):
    def __init__(self, data, full_data):
        self.data = data
        self.full_data = full_data
        self.len_data = len(data)
        self.len_full_data = len(full_data)

    def __getitem__(self, index):        
        geohashs = self.data[index]
        geohash_4_1, geohash_4_2 = random.sample(geohashs, 2)
        
        random_hashs = self.full_data[np.random.randint(0, self.len_full_data)]
        geohash_4_3 = random.sample(random_hashs, 1)[0]       
        return geohash_4_1, geohash_4_2, geohash_4_3

    def __len__(self):
        return self.len_data

class GeoHashModel(nn.Module):
    def __init__(self, embedding_dim, num_geohashes):
        super().__init__()
        self.embedding = nn.Embedding(num_geohashes, embedding_dim)

    def forward(self, geohash_indices):
        return self.embedding(geohash_indices) 

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.save_checkpoint(val_loss, model)
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        if self.verbose:
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

    def save_checkpoint(self, val_loss, model):
        print(f'Validation loss decreased ({self.best_score:.4f} ---> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)


geo_train = pd.read_parquet("Hackathon/geo_train.parquet")
geo_test = pd.read_parquet("Hackathon/geo_test.parquet")

geo = pd.concat([geo_train, geo_test])
print('Datasets red')
embedding_dim = 128
lr = 1e-3
loss_fn = torch.nn.TripletMarginLoss()
num_epochs = 30000
num_loss_round = 10
threshold = 0.002
for geohash in ['geohash_4', 'geohash_5']:
    # Try to use 2 keys of geohash
    for key_geohash in [True, False]:
        # Two ways of grouped data and embeddings creating. 
        if key_geohash:
            data = geo.groupby(['client_id', geohash]).count().reset_index()
            geohashes = data[geohash].unique()
            geohashes_dict = {int(par): i for i, par in enumerate(geohashes)}
            data[geohash] = data[geohash].map(geohashes_dict)                
            
            # Grouped by client to search for similarity of clients by geohashes
            client_dict = data.groupby('client_id')[geohash].agg([list, 'count']).reset_index()
            more = list(client_dict.loc[client_dict['count'] > 1, 'list'])
            full_data = list(client_dict['list'])
            shape = len(geohashes)
        else:
            data = geo.groupby(['client_id', geohash]).count().reset_index()
            count_hashs = data.groupby('client_id')[geohash].agg('count').reset_index()
            count_hashs.set_index('client_id', inplace = True)
            count_hashs = dict(count_hashs.iloc[:, 0])
            
            # Grouped by geohash to search for similarity of geohashes by clients
            count_client = data.groupby(geohash)['client_id'].agg([list, 'count']).reset_index()
            count_chashs = {}
            for i in tqdm(range(len(count_client))):
                for client in count_client.iloc[i, 1]:
                    count_chashs[client] = count_client.iloc[i, 2]
            
            clients = data['client_id'].unique()
            clients_dict = {par: i for i, par in enumerate(clients)}
            data['client_id'] = data['client_id'].map(clients_dict)
        
            client_dict = data.groupby(geohash)['client_id'].agg([list, 'count']).reset_index()
            more = list(client_dict.loc[client_dict['count'] > 1, 'list'])
            full_data = list(client_dict['list'])
            shape = len(clients)
        print('Grouped')
        model = GeoHashModel(embedding_dim, shape).cuda()
        
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, threshold = threshold, threshold_mode = 'abs', patience = 500)   
        dataset = GeoHashDataset(more, full_data)
        dataloader = DataLoader(dataset, batch_size = 2**14, num_workers = 0, drop_last = True, shuffle = True)

        writer = SummaryWriter()
        early_stopping = EarlyStopping(patience=1010, delta=threshold, verbose=True)
        all_losses = []
        for epoch in range(num_epochs):
            model.train()
            losses = []
            for batch_index, (geohash_4_1, geohash_4_2, geohash_4_3) in enumerate(dataloader):
                optimizer.zero_grad()
                
                geohash_4_1 = geohash_4_1.cuda().long()
                geohash_4_2 = geohash_4_2.cuda().long()
                geohash_4_3 = geohash_4_3.cuda().long()        
                
                anchor_embeddings = model(geohash_4_1)
                positive_embeddings = model(geohash_4_2)
                negative_embeddings = model(geohash_4_3)
        
                loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
        
            avg_bath_loss = np.mean(losses)
            all_losses.append(avg_bath_loss)
            if len(all_losses) <= num_loss_round:
                avg_loss = np.mean(all_losses)
            else:
                avg_loss = np.mean(all_losses[-num_loss_round:])
            print('Epoch [{}/{}] Loss: {:.4f} LR: {}'.format(epoch + 1, num_epochs, avg_loss, lr))
            
            lr_scheduler.step(avg_loss)
            lr = lr_scheduler._last_lr[0]
            writer.add_scalar('Epoch Loss', avg_loss, epoch)
            writer.add_scalar('Learning Rate', lr, epoch)
            
            early_stopping(val_loss = avg_loss, model=model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        writer.close()
        model.load_state_dict(torch.load('checkpoint.pt'))
        
        if key_geohash:
            def process_geo_hash(df):
                pandas_dict = {
                    "client_id": [],
                    "embedding": []
                }
                for client_id, rows in tqdm(df.groupby("client_id"), smoothing = 0):
                    unique_values = rows[geohash].unique()
                    filtered = [geohashes_dict[v] for v in unique_values if v in geohashes_dict]
                    if len(filtered) == 0:
                        avg_vec = np.zeros((embedding_dim, ))
                    else:
                        inputs = torch.from_numpy(np.array(filtered)).cuda().long()
                        avg_vec = model(inputs).mean(dim=0).detach().cpu().numpy()
                        
                    for month in range(1, 13):
                        pandas_dict["embedding"].append(avg_vec)
                        pandas_dict["client_id"].append(client_id + '_month=' + str(month))
                return pd.DataFrame.from_dict(pandas_dict)
        
            def process_geo_hash_test(df):
                pandas_dict = {
                    "client_id": [],
                    "embedding": []
                }
                for client_id, rows in tqdm(df.groupby("client_id"), smoothing = 0):
                    unique_values = rows[geohash].unique()
                    filtered = [geohashes_dict[v] for v in unique_values if v in geohashes_dict]
                    if len(filtered) == 0:
                        avg_vec = np.zeros((embedding_dim, ))
                    else:
                        inputs = torch.from_numpy(np.array(filtered)).cuda().long()
                        avg_vec = model(inputs).mean(dim=0).detach().cpu().numpy()
                        
                    pandas_dict["embedding"].append(avg_vec)
                    pandas_dict["client_id"].append(client_id)
                return pd.DataFrame.from_dict(pandas_dict)
            
            geo_train_df = process_geo_hash(geo_train)
            geo_validate_df = process_geo_hash(geo_test)
            geo_test_df = process_geo_hash_test(geo_test)
            
            cols = [f"geo_hash_{i}"  for i in range(embedding_dim)]
            geo_train_df[cols] = pd.DataFrame(geo_train_df["embedding"].tolist(), index=geo_train_df.index)
            geo_validate_df[cols] = pd.DataFrame(geo_validate_df["embedding"].tolist(), index=geo_validate_df.index)
            geo_test_df[cols] = pd.DataFrame(geo_test_df["embedding"].tolist(), index=geo_test_df.index)
            
            geo_train_df.drop("embedding", axis=1, inplace=True)
            geo_validate_df.drop("embedding", axis=1, inplace=True)
            geo_test_df.drop("embedding", axis=1, inplace=True)
            
            geo_train_df.to_parquet(f"created_data/train_{geohash}_hash.parquet", index=False, engine="pyarrow", compression="snappy")
            geo_validate_df.to_parquet(f"created_data/validate_{geohash}_hash.parquet", index=False, engine="pyarrow", compression="snappy")
            geo_test_df.to_parquet(f"created_data/test_{geohash}_hash.parquet", index=False, engine="pyarrow", compression="snappy")
        else:    
            def process_geo(df):
                clients = df['client_id'].unique()
                vec = []
                vec_clients = []
                vec_hashs = []
                for client in tqdm(clients, total = len(clients), smoothing = 0):
                    client_ind = clients_dict[client]
                    inputs = torch.tensor(client_ind).cuda().long()
                    vec_i = model(inputs).detach().cpu().numpy()
                    hashi = count_hashs[client]
                    chachi = count_chashs[client]                    
                    for month in range(1, 13): 
                        vec.append(vec_i)
                        vec_clients.append(client + '_month=' + str(month))
                        vec_hashs.append([hashi, chachi])
                cols = [f"{geohash}_client_{i}"  for i in range(embedding_dim)]
                emb_df = pd.concat([pd.DataFrame(vec_clients, columns = ['client_id']), pd.DataFrame(vec, columns = cols)], axis = 1)
                num_df = pd.concat([pd.DataFrame(vec_clients, columns = ['client_id']), pd.DataFrame(vec_hashs, columns = ['num_h', 'num_ch'])], axis = 1)
                return emb_df, num_df
            
            def process_geo_test(df):
                clients = df['client_id'].unique()
                vec = []
                vec_hashs = []
                for client in tqdm(clients, total = len(clients), smoothing = 0):
                    vec_hashs.append([count_hashs[client], count_chashs[client]])
                    client = clients_dict[client]
                    inputs = torch.tensor(client).cuda().long()
                    vec.append(model(inputs).detach().cpu().numpy())                    
                cols = [f"{geohash}_client_{i}"  for i in range(embedding_dim)]
                emb_df = pd.concat([pd.DataFrame(clients, columns = ['client_id']), pd.DataFrame(vec, columns = cols)], axis = 1)
                num_df = pd.concat([pd.DataFrame(clients, columns = ['client_id']), pd.DataFrame(vec_hashs, columns = ['num_h', 'num_ch'])], axis = 1)
                return emb_df, num_df
            
            geo_train_df, num_train_df = process_geo(geo_train)
            geo_validate_df, num_validate_df = process_geo(geo_test)
            geo_test_df, num_test_df = process_geo_test(geo_test)
            
            geo_train_df.to_parquet(f"created_data/train_{geohash}_client_20K.parquet", index=False, engine="pyarrow", compression="snappy")
            geo_validate_df.to_parquet(f"created_data/validate_{geohash}_client_20K.parquet", index=False, engine="pyarrow", compression="snappy")
            geo_test_df.to_parquet(f"created_data/test_{geohash}_client_20K.parquet", index=False, engine="pyarrow", compression="snappy")
            
            num_train_df.to_parquet(f"created_data/train_{geohash}_client_num.parquet", index=False, engine="pyarrow", compression="snappy")
            num_validate_df.to_parquet(f"created_data/validate_{geohash}_client_num.parquet", index=False, engine="pyarrow", compression="snappy")
            num_test_df.to_parquet(f"created_data/test_{geohash}_client_num.parquet", index=False, engine="pyarrow", compression="snappy")
        







