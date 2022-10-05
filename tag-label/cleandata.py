import os
import random
from tqdm import tqdm
import regex
import sys
import json
import multiprocessing


def restore(tokens, labels):
    res = ''
    index = 0
    while index < len(tokens):
        if labels[index].endswith('upper'):
            if tokens[index][0].islower():
                print("hello")
                res += tokens[index].capitalize() + " "
            else:
                res += tokens[index] + " "

        elif labels[index] == "O":
            res += tokens[index] + " "
        elif 'upper' in labels[index]:
            if tokens[index][0].islower():
                print("hello")
                res += tokens[index].capitalize() + labels[index].replace("upper"," ")+" "
            else:
                res += tokens[index] + labels[index].replace("upper"," ")+" "
        elif 'O' in labels[index]:
            res += tokens[index] + labels[index].replace("O"," ")+" "
        index += 1
    return res

def load_file(path):
    with open(path, "r", encoding="utf-8") as tmp:
        data = tmp.read()
    datas = data.split("\n\n")
    data = [sent.split("\n") for sent in datas ]
    data = [[word.split("\t") for word in sent] for sent in data]
    
    datas = []
    labels = []
    for sent in data:
        tmp_datas = []
        tmp_labels = []
        for word in sent:
            if len(word) < 2:
                print(sent)
                print(path)
                continue
            try:
                tmp_datas.append(word[0])
                tmp_labels.append(word[1])
            except:
                print(sent)
                print(path)
        datas.append(tmp_datas)
        labels.append(tmp_labels)
        
    sents = []
    for index in range(len(datas)):
        sent = restore(datas[index], labels[index])
        if sent == None:
            continue
        sents.append(sent)
    return sents

def preprocessing_datas(files):
    outpath = "/home/tuyendv/datas/datas/train"

    for file in tqdm(files):
        data = load_file(file)
        path = os.path.join(outpath, file.split("/")[-1])
        with open(path, "w", encoding="utf-8") as tmp:
            tmp.write("\n".join(data))
        print("saved: ", path)
        
        
path = "/home/tuyendv/datas/datas/train"
files = os.listdir(path)
files = [os.path.join(path, file) for file in files]

args = [tuple(files[i:i+50]) for i in range(0, len(files), 50)]
pool = multiprocessing.Pool(processes=len(args))

result_list = pool.map(preprocessing_datas, args)