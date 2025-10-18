import os, shutil, wget
import gzip
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import math

import torch

from torch_geometric.utils import coalesce
from torch_geometric.data import Data
from dataloader.utils import negative_sampling

class UCIDataset():
    def __init__(self, url, file_name, interval, train_ratio, val_ratio):
        super().__init__()
        self.url = url
        self.file_name = file_name
        self.time_interval = interval
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.download()
        self.preprocess()

        if self.time_interval in ['D', 'W', 'M']:
          self.make_graph_snapshot()
        else:
          self.split_by_seconds(self.time_interval)

        self.num_nodes = self.graph.node_feature.shape[0]
        self.num_edges = self.graph.edge_index.shape[1]
        self.num_snapshots = len(self.snapshot_list)

        self.snapshots = []
        for snapshot in self.snapshot_list:
            negative_edge_index = negative_sampling(snapshot.edge_index)
            snapshot.edge_label_index = torch.cat([snapshot.edge_index,
                                                   negative_edge_index], dim=-1)
            snapshot.edge_label = torch.cat([
                torch.ones(snapshot.edge_index.shape[1]),
                torch.zeros(negative_edge_index.shape[1])
            ])
            self.snapshots.append(snapshot)

        print('Total Snapshots: {}, Total Nodes: {}, Total Edges: {}'.format(self.num_snapshots,
                                                                             self.num_nodes, self.num_edges))

    def get_range_by_split(self, split):
        total_snapshots = len(self.snapshots)
        if split == 'train':
            begin = 0
            end = math.ceil(total_snapshots * self.train_ratio)
        elif split == 'val':
            begin = math.ceil(total_snapshots * self.train_ratio)
            end = math.ceil(total_snapshots * (self.train_ratio + self.val_ratio))
        elif split == 'test':
            begin = math.ceil(total_snapshots * (self.train_ratio + self.val_ratio))
            end = total_snapshots
        else:
            raise NotImplementedError('no such split {}'.format(split))
        return begin, end

    def download(self):
        if not os.path.exists('./data/{}'.format(self.file_name)):
            if not os.path.exists('./data'):
                os.mkdir('./data')
            print('Downloading ...')
            wget.download(self.url, out='./data')
            print('Done')
        else:
            print("File Already Downloaded")

    def preprocess(self):
        path = './data/{}'.format(self.file_name)
        with gzip.open(path, 'rb') as f_in:
            with open('./data/CollegeMsg.txt', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        path = './data/{}'.format('CollegeMsg.txt')
        df_trans = pd.read_csv(path, sep=' ', header=None)
        df_trans.columns = ['SRC', 'DST', 'TIMESTAMP']
        df_trans.reset_index(drop=True, inplace=True)

        # Node IDs of this dataset start from 1.
        df_trans['SRC'] -= 1
        df_trans['DST'] -= 1

        self.min_dst = min(df_trans['DST'])
        self.max_dst = max(df_trans['DST'])

        self.min_src = min(df_trans['SRC'])
        self.max_src = max(df_trans['SRC'])

        edge_index = torch.Tensor(df_trans[['SRC', 'DST']].values.transpose()).long()
        num_nodes = torch.max(edge_index) + 1

        node_feature = torch.rand(num_nodes, 64)

        node_id = torch.arange(num_nodes)

        edge_time = torch.FloatTensor(df_trans['TIMESTAMP'].values)

        self.graph = Data(node_feature=node_feature,
                           edge_index=edge_index,
                           edge_time=edge_time)

    def split_by_seconds(self, freq_sec):
        split_criterion = self.graph.edge_time // freq_sec
        groups = torch.sort(torch.unique(split_criterion))[0]
        self.snapshot_list = list()
        for t in groups:
            period_members = (split_criterion == t)
            g = Data(node_feature=self.graph.node_feature,
                      edge_index=self.graph.edge_index[:, period_members]
                      )
            g.edge_index = coalesce(g.edge_index)

            if g.edge_index.shape[1] > 2:
              self.snapshot_list.append(g)

    def make_graph_snapshot(self):
        t = self.graph.edge_time.numpy().astype(np.int64)

        period_split = pd.DataFrame(
            {'Timestamp': t, 'TransactionTime': pd.to_datetime(t, unit='s')},
            index=range(len(self.graph.edge_time))
        )

        freq_map = {'D': '%j',  # day of year.
                    'W': '%W',  # week of year.
                    'M': '%m'}  # month of year.

        period_split['Year'] = period_split['TransactionTime'].dt.strftime('%Y').astype(int)

        period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(freq_map[self.time_interval]).astype(int)

        period2id = period_split.groupby(['Year', 'SubYearFlag']).indices
        periods = sorted(list(period2id.keys()))
        self.snapshot_list = list()
        t = 0
        for p in periods:
            period_members = period2id[p]

            g = Data(node_feature=self.graph.node_feature,
                      edge_index=self.graph.edge_index[:, period_members],
                      )

            g.edge_index = coalesce(g.edge_index)

            if g.edge_index.shape[1] > 2:
              g.edge_time = torch.full((g.edge_index.shape[1], ), t + 1)
              t = t + 1
              self.snapshot_list.append(g)

class SXDataset():
    def __init__(self, url, file_name, interval, train_ratio, val_ratio):
        super().__init__()
        self.url = url
        self.file_name = file_name
        self.time_interval = interval
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.download()
        self.preprocess()

        if self.time_interval in ['D', 'W', 'M']:
          self.make_graph_snapshot()
        else:
          self.split_by_seconds(self.time_interval)

        self.num_nodes = self.graph.node_feature.shape[0]
        self.num_edges = self.graph.edge_index.shape[1]
        self.num_snapshots = len(self.snapshot_list)

        self.snapshots = []
        for snapshot in self.snapshot_list:
            negative_edge_index = negative_sampling(snapshot.edge_index)
            snapshot.edge_label_index = torch.cat([snapshot.edge_index,
                                                   negative_edge_index], dim=-1)
            snapshot.edge_label = torch.cat([
                torch.ones(snapshot.edge_index.shape[1]),
                torch.zeros(negative_edge_index.shape[1])
            ])
            self.snapshots.append(snapshot)

        print('Total Snapshots: {}, Total Nodes: {}, Total Edges: {}'.format(self.num_snapshots,
                                                                             self.num_nodes, self.num_edges))

    def get_range_by_split(self, split):
        total_snapshots = len(self.snapshots)
        if split == 'train':
            begin = 0
            end = math.ceil(total_snapshots * self.train_ratio)
        elif split == 'val':
            begin = math.ceil(total_snapshots * self.train_ratio)
            end = math.ceil(total_snapshots * (self.train_ratio + self.val_ratio))
        elif split == 'test':
            begin = math.ceil(total_snapshots * (self.train_ratio + self.val_ratio))
            end = total_snapshots
        else:
            raise NotImplementedError('no such split {}'.format(split))
        return begin, end

    def download(self):
        if not os.path.exists('./data/{}'.format(self.file_name)):
            if not os.path.exists('./data'):
                os.mkdir('./data')
            print('Downloading ...')
            wget.download(self.url, out='./data')
            print('Done')
        else:
            print("File Already Downloaded")

    def preprocess(self):
        path = './data/{}'.format(self.file_name)
        with gzip.open(path, 'rb') as f_in:
            with open('./data/stack-exchange.txt'.format(self.file_name), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        path = './data/{}'.format('stack-exchange.txt')
        df_trans = pd.read_csv(path, sep=' ', header=None)
        df_trans.columns = ['SOURCE', 'TARGET', 'TIMESTAMP']
        df_trans.reset_index(drop=True, inplace=True)

        node_indices = np.sort(pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))
        enc = OrdinalEncoder(categories=[node_indices, node_indices])
        raw_edges = df_trans[['SOURCE', 'TARGET']].values
        edge_index = enc.fit_transform(raw_edges).transpose()
        edge_index = torch.LongTensor(edge_index)

        num_nodes = len(pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))

        node_feature = torch.rand(num_nodes, 64)

        node_id = torch.arange(num_nodes)

        edge_time = torch.FloatTensor(df_trans['TIMESTAMP'].values)

        self.graph = Data(node_feature=node_feature,
                           edge_index=edge_index,
                           edge_time=edge_time)

    def split_by_seconds(self, freq_sec):
        split_criterion = self.graph.edge_time // freq_sec
        groups = torch.sort(torch.unique(split_criterion))[0]
        self.snapshot_list = list()
        for t in groups:
            period_members = (split_criterion == t)
            g = Data(node_feature=self.graph.node_feature,
                      edge_index=self.graph.edge_index[:, period_members]
                      )

            g.edge_index = coalesce(g.edge_index)

            if g.edge_index.shape[1] > 2:
              self.snapshot_list.append(g)

    def make_graph_snapshot(self):
        t = self.graph.edge_time.numpy().astype(np.int64)

        period_split = pd.DataFrame(
            {'Timestamp': t, 'TransactionTime': pd.to_datetime(t, unit='s')},
            index=range(len(self.graph.edge_time))
        )

        freq_map = {'D': '%j',  # day of year.
                    'W': '%W',  # week of year.
                    'M': '%m'}  # month of year.

        period_split['Year'] = period_split['TransactionTime'].dt.strftime('%Y').astype(int)

        period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(freq_map[self.time_interval]).astype(int)

        period2id = period_split.groupby(['Year', 'SubYearFlag']).indices
        periods = sorted(list(period2id.keys()))
        self.snapshot_list = list()
        t = 0
        for p in periods:
            period_members = period2id[p]

            g = Data(node_feature=self.graph.node_feature,
                      edge_index=self.graph.edge_index[:, period_members],
                      )

            g.edge_index = coalesce(g.edge_index)

            if g.edge_index.shape[1] > 2:
              g.edge_time = torch.full((g.edge_index.shape[1], ), t + 1)
              t = t + 1
              self.snapshot_list.append(g)

class BitCoinDataset():
    def __init__(self, url, file_name, interval, train_ratio, val_ratio):
        super().__init__()
        self.url = url
        self.file_name = file_name
        self.time_interval = interval
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.download()
        self.preprocess()

        if self.time_interval in ['D', 'W', 'M']:
          self.make_graph_snapshot()
        else:
          self.split_by_seconds(self.time_interval)

        self.num_nodes = self.graph.node_feature.shape[0]
        self.num_edges = self.graph.edge_index.shape[1]
        self.num_snapshots = len(self.snapshot_list)

        self.snapshots = []
        for snapshot in self.snapshot_list:
            negative_edge_index = negative_sampling(snapshot.edge_index)
            snapshot.edge_label_index = torch.cat([snapshot.edge_index,
                                                   negative_edge_index], dim=-1)
            snapshot.edge_label = torch.cat([
                torch.ones(snapshot.edge_index.shape[1]),
                torch.zeros(negative_edge_index.shape[1])
            ])
            self.snapshots.append(snapshot)

        print('Total Snapshots: {}, Total Nodes: {}, Total Edges: {}'.format(self.num_snapshots,
                                                                             self.num_nodes, self.num_edges))

    def download(self):
        if not os.path.exists('./data/{}'.format(self.file_name)):
            if not os.path.exists('./data'):
                os.mkdir('./data')
            print('Downloading ...')
            wget.download(self.url, out='./data')
            print('Done')
        else:
            print("File Already Downloaded")

    def get_range_by_split(self, split):
        total_snapshots = len(self.snapshots)
        if split == 'train':
            begin = 0
            end = math.ceil(total_snapshots * self.train_ratio)
        elif split == 'val':
            begin = math.ceil(total_snapshots * self.train_ratio)
            end = math.ceil(total_snapshots * (self.train_ratio + self.val_ratio))
        elif split == 'test':
            begin = math.ceil(total_snapshots * (self.train_ratio + self.val_ratio))
            end = total_snapshots
        else:
            raise NotImplementedError('no such split {}'.format(split))
        return begin, end

    def preprocess(self):
        path = './data/{}'.format(self.file_name)
        with gzip.open(path, 'rb') as f_in:
            with open('./data/soc-sign-bitcoin.csv', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        path = './data/{}'.format('soc-sign-bitcoin.csv')
        df_trans = pd.read_csv(path, sep=',', header=None, index_col=None)

        df_trans.columns = ['SOURCE', 'TARGET', 'RATING', 'TIME']

        node_indices = np.sort(pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))
        enc = OrdinalEncoder(categories=[node_indices, node_indices])
        raw_edges = df_trans[['SOURCE', 'TARGET']].values

        edge_index = enc.fit_transform(raw_edges).transpose()
        edge_index = torch.LongTensor(edge_index)

        self.min_src = min(edge_index[0]).item()
        self.max_src = max(edge_index[0]).item()

        self.min_dst = min(edge_index[1]).item()
        self.max_dst = max(edge_index[1]).item()

        num_nodes = len(pd.unique(df_trans[['SOURCE', 'TARGET']].to_numpy().ravel()))

        node_feature = torch.rand(num_nodes, 64)

        edge_time = torch.FloatTensor(df_trans['TIME'].values)

        self.graph = Data(node_feature=node_feature,
                           edge_index=edge_index,
                           edge_time=edge_time)

    def split_by_seconds(self, freq_sec):
        split_criterion = self.graph.edge_time // freq_sec
        groups = torch.sort(torch.unique(split_criterion))[0]
        self.snapshot_list = list()
        for t in groups:
            period_members = (split_criterion == t)
            g = Data(node_feature=self.graph.node_feature,
                      edge_index=self.graph.edge_index[:, period_members],
                      edge_time=self.graph.edge_time[period_members])
            if g.edge_index.shape[1] >= 10:
              self.snapshot_list.append(g)

    def make_graph_snapshot(self):
        t = self.graph.edge_time.numpy().astype(np.int64)

        period_split = pd.DataFrame(
            {'Timestamp': t, 'TransactionTime': pd.to_datetime(t, unit='s')},
            index=range(len(self.graph.edge_time))
        )

        freq_map = {'D': '%j',  # day of year.
                    'W': '%W',  # week of year.
                    'M': '%m'}  # month of year.

        period_split['Year'] = period_split['TransactionTime'].dt.strftime('%Y').astype(int)

        period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(freq_map[self.time_interval]).astype(int)

        period2id = period_split.groupby(['Year', 'SubYearFlag']).indices
        periods = sorted(list(period2id.keys()))
        self.snapshot_list = list()
        t = 0
        for p in periods:
            period_members = period2id[p]

            g = Data(node_feature=self.graph.node_feature,
                      edge_index=self.graph.edge_index[:, period_members],
                      )

            g.edge_index = coalesce(g.edge_index)

            if g.edge_index.shape[1] >= 10:
              g.edge_time = torch.full((g.edge_index.shape[1], ), t + 1)
              t = t + 1
              self.snapshot_list.append(g)

class TechDataset():
    def __init__(self, file_name, interval, train_ratio, val_ratio):
        super().__init__()
        self.file_name = file_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.time_interval = interval
        self.preprocess()

        self.make_graph_snapshot()

        self.num_nodes = self.graph.node_feature.shape[0]
        self.num_edges = self.graph.edge_index.shape[1]
        self.num_snapshots = len(self.snapshot_list)

        self.snapshots = []
        for snapshot in self.snapshot_list:
            negative_edge_index = negative_sampling(snapshot.edge_index)
            snapshot.edge_label_index = torch.cat([snapshot.edge_index,
                                                   negative_edge_index], dim=-1)
            snapshot.edge_label = torch.cat([
                torch.ones(snapshot.edge_index.shape[1]),
                torch.zeros(negative_edge_index.shape[1])
            ])
            self.snapshots.append(snapshot)

        print('Total Snapshots: {}, Total Nodes: {}, Total Edges: {}'.format(self.num_snapshots,
                                                                             self.num_nodes, self.num_edges))

    def get_range_by_split(self, split):
        total_snapshots = len(self.snapshots)
        if split == 'train':
            begin = 0
            end = math.ceil(total_snapshots * self.train_ratio)
        elif split == 'val':
            begin = math.ceil(total_snapshots * self.train_ratio)
            end = math.ceil(total_snapshots * (self.train_ratio + self.val_ratio))
        elif split == 'test':
            begin = math.ceil(total_snapshots * (self.train_ratio + self.val_ratio))
            end = total_snapshots
        else:
            raise NotImplementedError('no such split {}'.format(split))
        return begin, end

    def preprocess(self):
        df = pd.read_csv(self.file_name, sep=' ', header=None)
        df.columns = ['src', 'dst', 'w', 'ts']
        df.reset_index(drop=True, inplace=True)
        df['src'] -= 1
        df['dst'] -= 1

        edge_index = torch.Tensor(df[['src', 'dst']].values.transpose()).long()  # (2, E)
        num_nodes = torch.max(edge_index) + 1

        node_feature = torch.rand(num_nodes, 64)
        edge_time = torch.FloatTensor(df['ts'].values)

        self.graph = Data(
            node_feature=node_feature,
            edge_index=edge_index,
            edge_time=edge_time,
        )
        print(self.graph)

    def make_graph_snapshot(self):
        t = self.graph.edge_time.numpy().astype(np.int64)

        period_split = pd.DataFrame(
            {'Timestamp': t, 'TransactionTime': pd.to_datetime(t, unit='s')},
            index=range(len(self.graph.edge_time))
        )

        freq_map = {'D': '%j',  # day of year.
                    'W': '%W',  # week of year.
                    'M': '%m'}  # month of year.

        period_split['Year'] = period_split['TransactionTime'].dt.strftime('%Y').astype(int)

        period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(freq_map[self.time_interval]).astype(int)

        period2id = period_split.groupby(['Year', 'SubYearFlag']).indices
        periods = sorted(list(period2id.keys()))
        self.snapshot_list = list()
        t = 0
        for p in periods:
            period_members = period2id[p]

            g = Data(node_feature=self.graph.node_feature,
                      edge_index=self.graph.edge_index[:, period_members],
                      )

            g.edge_index = coalesce(g.edge_index)

            if g.edge_index.shape[1] > 2:
              g.edge_time = torch.full((g.edge_index.shape[1], ), t + 1)
              t = t + 1
              self.snapshot_list.append(g)