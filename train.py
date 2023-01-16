import torch
import numpy as np

import copy
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors
from memory_profiler import profile

import torch
from transformers import logging

import dataset
import util
import models
import eval

logging.set_verbosity_error()


N_EPOCH = 50
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

EARLY_STOP = 10

def train():

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


  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

  all_cells_tensor_train = get_cells(GRAPH_EMBEDDING_TRAIN, train_loader, device)
  all_cells_tensor_test = get_cells(GRAPH_EMBEDDING_TEST, test_loader, device)
  all_cells_tensor_dev = get_cells(GRAPH_EMBEDDING_DEV, dev_loader, device)


  model.train()

  n_cells_train = len(train_loader.dataset.cellid_to_label)
  n_cells_test = len(test_loader.dataset.cellid_to_label)
  n_cells_dev = len(dev_loader.dataset.cellid_to_label)

  best_error_distances = np.inf
  early_stop_counter = 0
  for epoch_idx in range(0, N_EPOCH):
    model.train()
    running_loss = 0
    for _, batch in enumerate(train_loader):
      optimizer.zero_grad()

      text = {key: val.to(device) for key, val in batch['text'].items()}

      target = batch['label'].to(device)

      loss, _ = model(text, target, all_cells_tensor_train)

      running_loss += loss
    
      loss.backward()

      optimizer.step()

    print (f"Finished training epoch {epoch_idx}, Loss: {running_loss}")

    model.eval()
    true_polygon_list, pred_points_list = [], []

    for _, batch in enumerate(dev_loader):

      text = {key: val.to(device) for key, val in batch['text'].items()}

      target = batch['label'].to(device)

      _, score = model(text, target, all_cells_tensor_dev)

      predictions = np.argmax(score.detach().cpu().numpy(), axis=-1)

      pred_points = util.predictions_to_points(predictions, dev_loader.dataset.label_to_cellid)
      pred_points_list.append(pred_points)

      true_polygon_list += [poly for poly in batch['geometry'][0]]

    pred_points_list = np.concatenate(pred_points_list, axis=0)
    error_distances_list, error_distances = eval.get_error_distances(true_polygon_list, pred_points_list)
    if best_error_distances > error_distances:
      best_error_distances = error_distances
      print (f"Found a better model with mean distance error {error_distances}")
      best_model = copy.deepcopy(model)
      print ("Evaluationg development set")
      eval.compute_metrics(error_distances_list)
    else:
      early_stop_counter += 1
    
    if early_stop_counter >= EARLY_STOP:
      print ("Early stopping")
      break
  
  del model
  best_model.eval()
  print ("Evaluationg test set")

  true_polygon_list, pred_points_list = [], []

  for _, batch in enumerate(test_loader):

    text = {key: val.to(device) for key, val in batch['text'].items()}

    target = batch['label'].to(device)

    _, score = best_model(text, target, all_cells_tensor_test)

    predictions = np.argmax(score.detach().cpu().numpy(), axis=-1)

    pred_points = util.predictions_to_points(predictions, test_loader.dataset.label_to_cellid)
    pred_points_list.append(pred_points)

    true_polygon_list += [poly for poly in batch['geometry'][0]]

  pred_points_list = np.concatenate(pred_points_list, axis=0)
  error_distances_list, _ = eval.get_error_distances(true_polygon_list, pred_points_list)
  eval.compute_metrics(error_distances_list)

def get_cells(graph_embedding_path, data_loader, device):
  
  graph_embed_file = KeyedVectors.load_word2vec_format(graph_embedding_path)

  all_cells = [graph_embed_file[str(c)] for c in data_loader.dataset.cellid_to_label.keys()]

  all_cells_tensor = torch.from_numpy(np.array(all_cells)).to(device)

  return all_cells_tensor


def check_grad(model):
  list_not_learned = []
  list_learned = []
  for name, param in model.named_parameters():
    if param.grad is None or param.grad.float().sum().tolist() == 0:
      list_not_learned.append(name)
    if param.grad is not None:
      list_learned.append(name)

  print(f"{len(list_not_learned)} Un-Learned params: {','.join(list_not_learned)}")

if __name__ == "__main__":

    train()