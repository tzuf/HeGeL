
from transformers import BertModel
import torch.nn as nn
import torch

BERT_TYPE = 'onlplab/alephbert-base'

criterion = torch.nn.CrossEntropyLoss()


s2cell_dim = 64
output_dim = 768
class DualEncoder(nn.Module):
  def __init__(self):
    super(DualEncoder, self).__init__()
    self.bert_model = BertModel.from_pretrained(
      BERT_TYPE, return_dict=True)

    self.cellid_main = nn.Sequential(
      nn.Linear(s2cell_dim, output_dim),
    )

  def forward(self, text, target, all_cells_tensor_test):
   
    cell = self.cellid_main(all_cells_tensor_test)
    encoded_text = self.bert_model(**text).last_hidden_state[:, 0, :] 
    dim_batch = target.shape[0]
    dim_cell = cell.shape[0]
    encoded_text_exp = encoded_text.unsqueeze(1).expand(dim_batch, dim_cell, output_dim)
    sample_cell_exp = cell.unsqueeze(0).expand(dim_batch, dim_cell, output_dim)

    score = torch.nn.functional.cosine_similarity(encoded_text_exp, sample_cell_exp, dim =-1)

    loss = criterion(score, target)

    return loss, score

