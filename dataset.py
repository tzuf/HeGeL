import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
from sklearn.utils import shuffle
from transformers import BertTokenizer

import regions
import util

BERT_TYPE = 'onlplab/alephbert-base'
unique_cellid_path = Path('Data/bauman_repres/unique_cellid.npy')
cellids_vectors_path = Path('Data/bauman_repres/cellids_vectors.npy')

text_tokenizer = BertTokenizer.from_pretrained(BERT_TYPE)


class HeGeLSplit(torch.utils.data.Dataset):
    """A split of of the HeGeL dataset.
  """

    def __init__(
            self,
            region: str,
            data_dir: str,
            split_set: str,
            graph_embed_path: str = None,
    ):
        # self.graph_embed_file = KeyedVectors.load_word2vec_format(graph_embed_path)
        # first_cell = self.graph_embed_file.index_to_key[0]
        # self.graph_embed_size = self.graph_embed_file[first_cell].shape[0]
        #
        # print(f"Dataset {split_set} with graph embedding size {self.graph_embed_size}")

        self.data = self.load_data(data_dir, split_set, lines=True)
        active_region = regions.get_region(region)
        # todo add unique cellid
        self.unique_cellid = np.load(str(unique_cellid_path))

        self.label_to_cellid = {idx: cellid for idx, cellid in enumerate(self.unique_cellid)}
        self.cellid_to_label = {cellid: idx for idx, cellid in enumerate(self.unique_cellid)}

        # todo get cellid for all the endpoints
        self.data.cellid =  pd.Series(self.data.goal_point_index.tolist())

        self.geometry = self.data.geometry.tolist()

        self.text_encodings = text_tokenizer(self.data.instructions.tolist(), truncation=True, padding=True,
                                             add_special_tokens=True, max_length=200)

        self.labels = self.data.cellid.apply(lambda x: self.cellid_to_label[x]).tolist()

    def load_data(self, data_dir: str, ds_set: str, lines: bool):
        ds_path = os.path.join(data_dir, ds_set + '.json')
        assert os.path.exists(ds_path), f"{ds_path} doesn't exsits"

        ds = pd.read_json(ds_path, lines=True)
        ds['instructions'] = ds['content']
        ds['end_point'] = ds['goal_point'].apply(util.point_from_str_coord_xy)
        ds['geometry'] = ds['geometry'].apply(util.list_arrays_from_str_geometry)
        ds['goal_point_index'] = ds['true_polygon_indexes']
        ds = shuffle(ds)
        ds.reset_index(inplace=True, drop=True)
        return ds

    def __getitem__(self, idx: int):
        '''Supports indexing such that TextGeoDataset[i] can be used to get
    i-th sample.
    Arguments:
      idx: The index for which a sample from the dataset will be returned.
    Returns:
      A single sample including text, the correct cellid, a neighbor cellid,
      a far cellid, a point of the cellid and the label of the cellid.
    '''

        text_input = {key: torch.tensor(val[idx])
                      for key, val in self.text_encodings.items()}

        labels = torch.tensor(self.labels[idx])

        sample = {'text': text_input,
                  'label': labels,
                  'geometry': [self.geometry[idx]],
                  }

        return sample

    def __len__(self):
        return len(self.data.cellid)
