from src.resources import hparams
from src.trainer import trainer
from typing import Union
import torch

if __name__ == "__main__":
    mode = hparams.mode
    print("mode: ", mode)
    
    if mode == "train":
        cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("cuda: ", cuda)
        trainer = trainer(cuda=cuda, mode=mode, is_warm_up=True, bert_type=hparams.bert)
        trainer.train()
    elif mode == "test":
        cuda = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
        print("cuda: ", cuda)
        trainer = trainer(cuda=cuda, mode=mode, bert_type=hparams.bert)
    
        trainer.run_test()
