from src.resources import hparams
from src.trainer import trainer
from typing import Union
import torch

if __name__ == "__main__":
    mode = hparams.mode
    print("mode: ", mode)
    
    if mode == "train":
        cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(cuda)
        trainer = trainer(cuda=cuda, mode=mode, is_warm_up=True, bert=hparams.bert)
        trainer.train()
    elif mode == "test":
        cuda = torch.device(hparams.cuda) if torch.cuda.is_available() else 'cpu'
        print(cuda)
        trainer = trainer(cuda=cuda, mode=mode, bert=hparams.bert)
    
        trainer.run_test()