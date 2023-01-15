import torch
import numpy as np

from torch.utils.data import DataLoader
from gensim.models import KeyedVectors

from transformers import AdamW


import torch

import dataset
import util
import models
import eval

N_EPOCH = 30
S2_LEVEL = 13
DATA_DIR = 'data/human'

REGION_TRAIN = 'Tel_Aviv'
REGION_DEV = 'Haifa'
REGION_TEST = 'Jerusalem'

GRAPH_EMBEDDING_TRAIN = f'data/cell_embedding/embedding_tel_aviv_{S2_LEVEL}.npy'
GRAPH_EMBEDDING_DEV = f'data/cell_embedding/embedding_haifa_{S2_LEVEL}.npy'
GRAPH_EMBEDDING_TEST = f'data/cell_embedding/embedding_jerusalem_{S2_LEVEL}.npy'

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
DEV_BATCH_SIZE = 32

dataset_train = dataset.HeGeLSplit(
  region=REGION_TRAIN, 
  data_dir=DATA_DIR, 
  split_set='train',
  s2level=S2_LEVEL,
  graph_embed_path=GRAPH_EMBEDDING_TRAIN, 
  )

dataset_dev = dataset.HeGeLSplit(
  region=REGION_DEV, 
  data_dir=DATA_DIR, 
  split_set='dev',
  s2level=S2_LEVEL,
  graph_embed_path=GRAPH_EMBEDDING_DEV, 
  )

dataset_test = dataset.HeGeLSplit(
  region=REGION_TEST, 
  data_dir=DATA_DIR, 
  split_set='test',
  s2level=S2_LEVEL,
  graph_embed_path=GRAPH_EMBEDDING_TEST, 
  )

train_loader = DataLoader(
  dataset_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(
  dataset_dev, batch_size=TEST_BATCH_SIZE, shuffle=False)
test_loader = DataLoader(
  dataset_test, batch_size=DEV_BATCH_SIZE, shuffle=False)



device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


model = models.DualEncoder()


model.to(device)


optimizer = AdamW(model.parameters(), lr=1e-5)



def check_grad(model):
  list_not_learned = []
  list_learned = []
  for name, param in model.named_parameters():
    if param.grad is None or param.grad.float().sum().tolist() == 0:
      list_not_learned.append(name)
    if param.grad is not None:
      list_learned.append(name)

  print(f"{len(list_not_learned)} Un-Learned params: {','.join(list_not_learned)}")


graph_embed_train_file = KeyedVectors.load_word2vec_format(GRAPH_EMBEDDING_TRAIN)

cellids_train = train_loader.dataset.cellid_to_label.keys()

cellid_to_label = test_loader.dataset.cellid_to_label
label_to_cellid = {v: k for k,v in cellid_to_label.items()}

sample_cell_basic_train = []
for c in train_loader.dataset.cellid_to_label.keys():

  sample_cell_basic_train.append(graph_embed_train_file[str(c)])


sample_cell_basic_train = torch.from_numpy(np.array(sample_cell_basic_train)).to(device)



graph_embed_test_file = KeyedVectors.load_word2vec_format(GRAPH_EMBEDDING_TEST)


cellids_test = test_loader.dataset.cellid_to_label.keys()

sample_cell_basic_test = []
for c in test_loader.dataset.cellid_to_label.keys():

  sample_cell_basic_test.append(graph_embed_test_file[str(c)])


sample_cell_basic_test = torch.from_numpy(np.array(sample_cell_basic_test)).to(device)


model.train()

cer = torch.nn.CrossEntropyLoss()

n_cells_train = len(train_loader.dataset.cellid_to_label)
n_cells_test = len(test_loader.dataset.cellid_to_label)

for epoch_idx in range(0, N_EPOCH):
  model.train()
  running_loss = 0
  for batch_idx, batch in enumerate(train_loader):
    optimizer.zero_grad()
    sample_cell = model.cellid_main(sample_cell_basic_train)

    text = {key: val.to(device) for key, val in batch['text'].items()}

    target = batch['label'].to(device)


    encoded = model.bert_model(**text).last_hidden_state[:, 0, :] 
    dim_batch = target.shape[0]
    encoded_exp = encoded.unsqueeze(1).expand(dim_batch, n_cells_train, 768)
    sample_cell_exp = sample_cell.unsqueeze(0).expand(dim_batch, n_cells_train, 768)


    score = torch.nn.functional.cosine_similarity(encoded_exp, sample_cell_exp, dim =-1)

    loss = cer(score, target)

    running_loss += loss
  
    loss.backward()

    optimizer.step()

  print (f"Finished training epoch {epoch_idx}, Loss: {running_loss}")

model.eval()
true_polygon_list, pred_points_list = [], []

for batch_idx, batch in enumerate(test_loader):
  sample_cell = model.cellid_main(sample_cell_basic_test)

  text = {key: val.to(device) for key, val in batch['text'].items()}

  target = batch['label'].to(device)

  encoded = model.bert_model(**text).last_hidden_state[:, 0, :] 
  dim_batch = target.shape[0]
  encoded_exp = encoded.unsqueeze(1).expand(dim_batch, n_cells_test, 768)
  sample_cell_exp = sample_cell.unsqueeze(0).expand(dim_batch, n_cells_test, 768)

  score = torch.nn.functional.cosine_similarity(encoded_exp, sample_cell_exp, dim =-1).detach().cpu().numpy()
  predictions = np.argmax(score, axis=-1)

  pred_points = util.predictions_to_points(predictions, label_to_cellid)
  pred_points_list.append(pred_points)

  geometry_list = [poly for poly in batch['geometry'][0]]
  true_polygon_list += geometry_list

pred_points_list = np.concatenate(pred_points_list, axis=0)
error_distances = eval.get_error_distances(true_polygon_list, pred_points_list)
eval.compute_metrics(error_distances)



