import os
import random
from tqdm import tqdm
import regex
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
    data = [sent.split("\n") for sent in datas]
    data = [[word.split("\t") for word in sent] for sent in data]
    
    datas = []
    labels = []
    for sent in data:
        tmp_datas = []
        tmp_labels = []
        for word in sent:
            tmp_datas.append(word[0])
            tmp_labels.append(word[1])
        datas.append(tmp_datas)
        labels.append(tmp_labels)
        
    sents = []
    for index in range(len(datas)):
        sent = restore(datas[index], labels[index])
        sents.append(sent)
    return sents

def augment_data(files):
    datas=[]
    for file in tqdm(files):
        data = load_file(file)
        datas += data
    return datas

def filter(sent):
    if sent.count(",") > 2 and regex.search("\s[A-Z]{3}\s", sent) != None:
        return True
    return False

exotic = json.load(open("src/resources/dict/foreign.json", "r", encoding="utf-8"))
def replace(sent):
    for key, value in exotic.items():
        if " "+ key + " " in sent:
            sent = sent.replace(" "+ key + " ", " "+value[0]+" ")
    return sent

path = "/home/tuyendv/datas/text-recovery-data/train"
files = os.listdir(path)
files = [os.path.join(path, file) for file in files]
args = [tuple(files[i:i+40]) for i in range(0, len(files), 40)]
pool = multiprocessing.Pool(processes=len(args))

result_list = pool.map(augment_data, args)

datas = []
for data in result_list:
    datas += data