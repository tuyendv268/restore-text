import json
import pandas as pd
from sklearn.metrics import classification_report
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.model.bert_bilstm import bert_bilstm
from src.utils import *
from torch.utils.data import DataLoader
from src.dataset.dataset import Dataset
from src.resources import hparams
from torch import optim
from tqdm import tqdm
import numpy as np
import os
from torch import nn
import torch

class trainer():
    def __init__(self, cuda, is_warm_up=True, mode="train", infer_path=None, bert_type="envibert_cased"):
        self.device = cuda
        self.bert_type = bert_type
        
        self.model = bert_bilstm(
            cuda=cuda,
            nb_label=hparams.nb_labels, 
            bert_type=self.bert_type).to(cuda)
        self.tokenizer= self.model.tokenizer
          
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=hparams.lr, 
            weight_decay=hparams.weight_decay)
        
        print("Model Architecture: ", self.model)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=1)
        self.max_epoch = hparams.max_epoch
        self.is_warm_up = is_warm_up
        
        self.tag2index=json.load(open(hparams.tag2index,"r"))
        self.index2tag=json.load(open(hparams.index2tag,"r"))
        
        if mode == "train":
            if os.path.exists(hparams.warm_up) and is_warm_up == True:
                print("warm up: ", hparams.warm_up)
                self.model.load_state_dict(torch.load(hparams.warm_up, map_location=self.device))
                
            self.train_dl = self.load_data(path=hparams.train_path, 
                                           batch_size=hparams.train_bs, 
                                           tag2index=self.tag2index,
                                           max_sent_length=hparams.max_sent_length)
            self.val_dl = self.load_data(path=hparams.val_path, 
                                           batch_size=hparams.val_bs, 
                                           tag2index=self.tag2index,
                                           max_sent_length=hparams.max_sent_length)
            if hparams.parallel == True:
                self.model = nn.DataParallel(self.model).to(self.device)
        elif mode == "test":
            if os.path.exists(hparams.test_checkpoint):
                print("load model: ", hparams.test_checkpoint)
                self.model.load_state_dict(torch.load(hparams.test_checkpoint, map_location=self.device))
            else:
                print("Not exist! ")
            self.test_dl = self.load_data(path=hparams.test_path, 
                                           batch_size=hparams.test_bs, 
                                           tag2index=self.tag2index,
                                           max_sent_length=hparams.max_sent_length)
        elif mode == "infer":
            self.infer_path=infer_path
            
            if os.path.exists(self.infer_path):
                print("Load model: ", self.infer_path)
                self.model.load_state_dict(torch.load(self.infer_path, map_location="cpu"))
        
        elif mode == "demo":
            self.infer_path=infer_path
            
            if os.path.exists(self.infer_path):
                print("Load model: ", self.infer_path)
                self.model.load_state_dict(torch.load(self.infer_path, map_location=cuda))
                torch.save("demo.pt")
        
    def load_data(self, path, batch_size, tag2index,max_sent_length):
        print("load data: ", path)
        print("max sentence length: ", max_sent_length)
        
        input_ids, label_ids = load_data_parallel(
            path=path,
            tag2index=tag2index, 
            max_sent_length=max_sent_length
        )
        
        print("sample: ", input_ids[0])
        input_ids = torch.tensor(input_ids[:-1], dtype=torch.int32)
        print("shape: ", input_ids.shape)
        label_ids = torch.tensor(label_ids[:-1], dtype=torch.int32)
        
        self.data =  Dataset(input_ids=input_ids, 
                        label_ids=label_ids, 
                        max_sent_lenth=hparams.max_sent_length,
                        tokenizer=self.tokenizer,
                        tag2index=self.tag2index)
        
        return DataLoader(
            dataset=self.data, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=8)
            # pin_memory=True)
    
    def train(self):
        print("---------------start training---------------")
        for epoch in range(self.max_epoch):
            train_tqdm = tqdm(self.train_dl)
            self.model.train()
            for idx ,input_data in enumerate(train_tqdm):
                self.model.zero_grad()
                
                input_ids = input_data["input_ids"].to(self.device)
                input_masks = input_data["input_masks"].to(self.device)
                label_ids = input_data["label_ids"].to(self.device)
                label_masks = input_data["label_masks"].to(self.device)
                
                predict = self.model(input_ids, input_masks)
                
                loss = self.loss(predict.transpose(1,2), label_ids)
                
                train_tqdm.set_postfix({"epoch":epoch, "loss":loss.item()})
                loss.backward()
                self.optimizer.step()
                
                if((idx+1) % 20000 == 0):
                    PATH = hparams.checkpoint_path.replace("%EPOCH%", str(idx+1))
                    torch.save(self.model.state_dict(), PATH)
                    print("\nsaved checkpoint: ", PATH)
                
            # if ((epoch) % 1 == 0):
            PATH = hparams.checkpoint_path.replace("%EPOCH%", str(epoch))
            torch.save(self.model.to(hparams.val_cuda).state_dict(), PATH)
            print("\nsaved checkpoint: ", PATH)
            
            results,confus_matrix  = self.val(PATH)
                
            f1_path = hparams.res_path.replace("%EPOCH%", str(epoch))
            confusion_path = hparams.confusion_matrix_path.replace("%EPOCH%", str(epoch))
            
            self.save_confusion_matrix(confus_matrix, confusion_path)
            print("\nsaved: ", confusion_path)
            
            with open(f1_path, "w") as tmp:
                tmp.write(results)
                print("saved: ", f1_path)
        print("-------------------done------------------")
    
    def run_test(self):
        predicts, labels = None, None
        self.model.eval()
        test_tqdm = tqdm(self.test_dl)
        with torch.no_grad():
            for idx, input_data in enumerate(test_tqdm):
                input_ids = input_data["input_ids"].to(self.device)
                input_masks = input_data["input_masks"].to(self.device)
                label_ids = input_data["label_ids"].to(self.device)
                label_masks = input_data["label_masks"].to(self.device)

                predict = self.model(input_ids, input_masks)
                
                predict = torch.argmax(predict, dim = 2)
                            
                if (predicts ==None and labels == None):
                    predicts = predict.type(torch.int8)
                    labels = label_ids.type(torch.int8)
                else:
                    predicts = torch.cat((predicts, predict.type(torch.int8)), dim=0)
                    labels = torch.cat((labels, label_ids.type(torch.int8)), dim=0)
            predicts = predicts.view(-1)
            labels = labels.view(-1)
            target_names = [self.index2tag[str(i)] for i in range(hparams.nb_labels)]

            predicts = predicts.cpu().numpy()
            labels = labels.cpu().numpy()

            results = classification_report(y_pred=predicts,y_true=labels, target_names=target_names, labels=range(len(target_names)))
            confus_matrix = confusion_matrix(y_pred=predicts,y_true=labels, labels=range(len(target_names)))
            print("results: ", results)
            print("confus_matrix: ", confus_matrix)
        
            f1_path = hparams.res_path.replace("%EPOCH%", "test")
            confusion_path = hparams.confusion_matrix_path.replace("%EPOCH%", "test")
            
            self.save_confusion_matrix(confus_matrix, confusion_path)
            print("saved: ", confusion_path)
            
            with open(f1_path, "w") as tmp:
                tmp.write(results)
                print("saved: ", f1_path)
        
        return results
    
    def load_model(self, path, val_cuda):
        print("load checkpoint: ", path)
        model = bert_bilstm(
                    cuda=val_cuda,
                    nb_label=hparams.nb_labels, 
                    envibert=self.bert).to(val_cuda)  
        model.load_state_dict(torch.load(path, map_location=val_cuda))
        return model
    
    def val(self, path):
        with torch.no_grad():
            val_cuda = "cuda:"+str(hparams.val_cuda)
            model = self.model.to(val_cuda)
            predicts, labels = None, None
            
            model.eval()
            val_tqdm = tqdm(self.val_dl)
            for idx, input_data in enumerate(val_tqdm):
                input_ids = input_data["input_ids"].to(val_cuda)
                input_masks = input_data["input_masks"].to(val_cuda)
                label_ids = input_data["label_ids"].to(val_cuda)
                label_masks = input_data["label_masks"].to(val_cuda)

                predict = model(input_ids, input_masks)
                
                predict = torch.argmax(predict, dim = 2)
                            
                if (predicts ==None and labels == None):
                    predicts = predict.type(torch.int8)
                    labels = label_ids.type(torch.int8)
                else:
                    predicts = torch.cat((predicts, predict.type(torch.int8)), dim=0)
                    labels = torch.cat((labels, label_ids.type(torch.int8)), dim=0)
            predicts = predicts.view(-1)
            labels = labels.view(-1)
            # results = torchmetrics.functional.f1_score(preds=predicts, target=labels)
            target_names = [self.index2tag[str(i)] for i in range(hparams.nb_labels)]
            # print(target_names)
            predicts = predicts.cpu().numpy()
            labels = labels.cpu().numpy()
            # print(predicts.shape)
            # print(labels.shape)
            results = classification_report(y_pred=predicts,y_true=labels, target_names=target_names, labels=range(len(target_names)))
            confus_matrix = confusion_matrix(y_pred=predicts,y_true=labels, labels=range(len(target_names)))
            print(results)
            print(confus_matrix)
        return results, confus_matrix
    
    def save_confusion_matrix(self, confus_matrix, confusion_path):
        confus_matrix = np.array(confus_matrix, dtype=np.float64)
        nomalization= confus_matrix.sum(axis = 1, keepdims = True)
        nomalization[nomalization==0] = 1.
        print(nomalization)
        confus_matrix = confus_matrix/nomalization
        df_cm = pd.DataFrame(
            confus_matrix.round(2), 
            index = [self.index2tag[str(i)] for i in range(hparams.nb_labels)],
            columns = [self.index2tag[str(i)] for i in range(hparams.nb_labels)])
        plt.figure(figsize = (8,6))
        sn.heatmap(df_cm, annot=True, cmap="OrRd")
        plt.savefig(confusion_path)

    
    def infer(self, text, model, tokenizer):
        text = text.lower()
        input_ids, input_masks = prepare_data_for_infer(sent=text, tokenizer=tokenizer)

        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        input_ids = torch.tensor([input_ids]).to(self.device)
        input_masks = torch.tensor([input_masks]).to(self.device)

        pred = model(input_ids, input_masks)
        print(pred)
        pred = torch.argmax(pred, dim = 2).cpu().tolist()
        print(pred)
        
        join_tokens, join_tags = join(tokens, pred[0])
        
        join_tags = cvt_ids2label(join_tags, index2label=self.index2tag)
        print(join_tokens)
        print(join_tags)
        out_sent = self.restore(tokens=join_tokens, labels=join_tags)
        return out_sent

    def infer_sent(self, raw_text):
        inp_text = remove_punct(text=raw_text)
        out_text = self.infer(inp_text, self.model, self.tokenizer)
        return raw_text, out_text, inp_text
