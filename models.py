
from transformers import BertModel
import torch.nn as nn

BERT_TYPE = 'onlplab/alephbert-base'


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
