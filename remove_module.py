
from src.resources import hparams
from torch import nn
import torch

from importlib.machinery import SourceFileLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from transformers import RobertaModel
from fairseq.models.roberta import XLMRModel
from src.model.bilstm import BiLSTM
from src.resources import hparams
from torch import nn
import os

bert_type = "envibert_cased"

if bert_type =="envibert_cased":
    print("use: envibert")
    tokenizer = SourceFileLoader("envibert.tokenizer", 
            os.path.join(hparams.toknizer_path,'envibert_tokenizer.py')).load_module().RobertaTokenizer(hparams.toknizer_path)
    bert = RobertaModel.from_pretrained('nguyenvulebinh/envibert',cache_dir=hparams.pretrained_envibert_cased)
elif bert_type =="envibert_uncased":
    print("use: envibert_uncased")
    tokenizer = SourceFileLoader("envibert.tokenizer", 
            os.path.join(hparams.pretrained_envibert_uncased,'envibert_tokenizer.py')).load_module().RobertaTokenizer(hparams.pretrained_envibert_uncased)
    bert = XLMRModel.from_pretrained(hparams.pretrained_envibert_uncased, checkpoint_file='model.pt')
elif bert_type =="xlmr":
    print("use: xlm roberta")
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    bert = XLMRobertaModel.from_pretrained("xlm-roberta-base")

class bert_bilstm(nn.Module):
    def __init__(self, nb_label, cuda, bert, drop_rate, hidden_dim_lstm, hidden_dim_bert):
        super().__init__()
        self.bert = bert
        self.bilstm = BiLSTM(
            cuda=cuda, 
            embedding_dim=hidden_dim_bert, 
            hidden_dim=hidden_dim_lstm
            )
        self.dropout = nn.Dropout(drop_rate)
        self.linear = nn.Linear(hidden_dim_lstm, nb_label).to(cuda)

    def forward(self, input_ids, input_masks):
        output = self.bert(input_ids=input_ids, 
                        attention_mask = input_masks)
        sequence_output, _ = output[0], output[1]
        
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.bilstm(sequence_output)
        output = self.linear(sequence_output)

        return output

model = bert_bilstm(
        cuda="cuda:1",
        nb_label=hparams.nb_labels, 
        bert=bert,
        drop_rate=hparams.drop_rate,
        hidden_dim_bert=768,
        hidden_dim_lstm=hparams.hidden_dim_lstm).to("cuda:1")

tmp = torch.load("checkpoint_envibert_cased_1.pt")

res = {}
for key, value in tmp.items():
    res[key.replace("module.","")] = value
model.load_state_dict(res)

torch.save(model.state_dict(),"checkpoint_envibert_cased_1-removed-module.pt")
