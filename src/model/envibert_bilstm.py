from src.model import bilstm
from src.resources import hparams
from torch import nn

class envibert_bilstm(nn.Module):
    def __init__(self, nb_label, cuda, envibert):
        super().__init__()
        self.model = envibert
        self.bilstm = bilstm.BiLSTM(
            cuda=cuda, 
            emb_dim=hparams.embedding_dim, 
            hidden_dim=hparams.hidden_dim
            )
        self.dropout = nn.Dropout(hparams.drop_rate)
        self.linear = nn.Linear(hparams.hidden_dim, nb_label).to(cuda)

    def forward(self, input_ids, input_masks):
        output = self.model(input_ids=input_ids, 
                            attention_mask = input_masks)
        sequence_output, _ = output[0], output[1]
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.bilstm(sequence_output)
        output = self.linear(sequence_output)

        return output