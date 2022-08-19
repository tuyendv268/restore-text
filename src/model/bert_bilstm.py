from importlib.machinery import SourceFileLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from transformers import RobertaModel
from fairseq.models.roberta import XLMRModel
from src.model import bilstm
from src.resources import hparams
from torch import nn
import os

class bert_bilstm(nn.Module):
    def __init__(self, nb_label, cuda, bert_type):
        super().__init__()
        self.bert_type = bert_type
        if bert_type =="envibert_cased":
            print("use: envibert")
            self.tokenizer = SourceFileLoader("envibert.tokenizer", 
                    os.path.join(hparams.toknizer_path,'envibert_tokenizer.py')).load_module().RobertaTokenizer(hparams.toknizer_path)
            self.bert = RobertaModel.from_pretrained('nguyenvulebinh/envibert',cache_dir=hparams.pretrained_envibert_cased)
        elif bert_type =="envibert_uncased":
            print("use: envibert_uncased")
            self.tokenizer = SourceFileLoader("envibert.tokenizer", 
                    os.path.join(hparams.pretrained_envibert_uncased,'envibert_tokenizer.py')).load_module().RobertaTokenizer(hparams.pretrained_envibert_uncased)
            self.bert = XLMRModel.from_pretrained(hparams.pretrained_envibert_uncased, checkpoint_file='model.pt')
        elif bert_type =="xlmr":
            print("use: xlm roberta")
            self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
            self.bert = XLMRobertaModel.from_pretrained("xlm-roberta-base")
            
        self.bilstm = bilstm.BiLSTM(
            cuda=cuda, 
            embedding_dim=hparams.hidden_dim_bert, 
            hidden_dim=hparams.hidden_dim_lstm
            )
        self.dropout = nn.Dropout(hparams.drop_rate)
        self.linear = nn.Linear(hparams.hidden_dim_lstm, nb_label).to(cuda)

    def forward(self, input_ids, input_masks):
        if self.bert_type == "envibert_uncased":
            sequence_output = self.bert.extract_features(input_ids)
        elif self.bert_type == "envibert_cased":
            output = self.bert(input_ids=input_ids, 
                            attention_mask = input_masks)
            sequence_output, _ = output[0], output[1]
        elif self.bert_type == "xlmr":
            output = self.bert(input_ids, input_masks)
            sequence_output, _ = output[0], output[1]
        
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.bilstm(sequence_output)
        output = self.linear(sequence_output)

        return output