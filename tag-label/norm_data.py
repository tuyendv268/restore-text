import os
import random
from tqdm import tqdm
import regex
import json
import re
import multiprocessing

def load_json(path):
    with open(path, "r", encoding="utf-8") as tmp:
        data = json.load(tmp)
    return data

def load_resource():
    with open("stranger.txt","r", encoding="utf-8") as f:
        global EXOTIC
        EXOTIC = [ele.replace("\n","") for ele in f.readlines()]
        EXOTIC = set(EXOTIC)
    global ABBRE_V1, ABBRE_V2, FOREIGN_V1, FOREIGN_V2
    ABBRE_V1 = load_json("dictionary/abbre-v1.json")
    ABBRE_V2 = load_json("dictionary/abbre-v2.json") 
    FOREIGN_V1 = load_json("dictionary/foreign_v1.json")
    FOREIGN_V2 = load_json("dictionary/foreign_v2.json")
    
    
    
def load_file(path):
    with open(path, "r", encoding="utf-8") as tmp:
        sents = tmp.readlines()
        sents = [sent.replace("\n", "") for sent in sents]
    return sents

def is_ignore(sent):
    if regex.search("(?<=một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mốt|tư|lăm)\s+,\s+(?=một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mốt|tư|lăm)", sent) != None:
        return True

    for i in EXOTIC:
        if i in sent:
            # print("ignore: ", list(i))
            return True
    
    return False

def replace_upper_first(sent , dictionary):
    words = sent.split()
    n = len(words)
    for idx, word in enumerate(words):
        if not word[0].isupper():
            continue
        if word in dictionary:
            # print(word)
            if idx >=1 and idx < n-1:
                if words[idx-1] == "," and words[idx+1] == ",":
                    words[idx] = dictionary[word]
                if words[idx-1] == "," and words[idx+1][0].isupper():
                    continue
                if words[idx+1] == "," and words[idx-1][0].isupper():
                    continue
                if words[idx-1][0].isupper() or words[idx+1][0].isupper():
                    continue         
                words[idx] = dictionary[word]
    return " ".join(words)

def replace_upper(sent , dictionary):
    words = sent.split()
    for idx, word in enumerate(words):
        if not word.isupper():
            continue
        if word in dictionary:
            # print(word)
            words[idx] = dictionary[word]
    return " ".join(words)

def replace(sent , dictionary):
    for key in dictionary.keys():
        tmp = " "+ key+" "
        if tmp in sent:
            # print(tmp)
            if check(sent, key):
                sent = sent.replace(key, dictionary[key])
    return sent

def check(sent, key):
    index = sent.find(key)
    sent_length = len(sent)
    
    left_idx = index-2
    right_idx = index + len(key)
    if left_idx <= 0 or right_idx >= sent_length:
        return True
    
    while(sent[left_idx] != " "):
        left_idx -= 1
        if left_idx <= 0:
            break
        
    while(sent[right_idx] != " "):
        right_idx += 1
        if right_idx >= sent_length:
            break

    left_token = sent[left_idx+1:index]
    right_token = sent[index + len(key):right_idx]
    
    if len(left_token) < 1 or len(right_token) < 1:
        return False
    
    if left_token == "," and right_token == ",":
        return True
    if left_token == "," and right_token[0].isupper():
        return False
    if right_token == "," and left_token[0].isupper():
        return False

    if left_token[0].isupper() or right_token[0].isupper():
        return False
    return True

def norm_data(sent):
    sent = re.sub(", và ,", " và ", sent)
    sent = re.sub(", và ", " và " ,sent)
    
    sent = replace_upper_first(sent=sent, dictionary=FOREIGN_V1)
    sent = replace(sent=sent, dictionary=FOREIGN_V2)
      
    sent = replace_upper(sent=sent, dictionary=ABBRE_V1)
    sent = replace_upper(sent=sent, dictionary=ABBRE_V2)
    
    return sent
    
def preprocessing_datas(files):
    outpath = "text-restoration-data-new/test"

    for file in tqdm(files):
        sents = []
        if "subdata" not in file:
            print("ignore:", file)
            continue
        data = load_file(file)
        count = 0
        for sent in data:
            if is_ignore(sent):
                count +=1
                continue
            sent = norm_data(sent)
            sents.append(sent)
        print("ignore: ", count)
        path = os.path.join(outpath, file.split("/")[-1])
        with open(path, "w", encoding="utf-8") as tmp:
            tmp.write("\n".join(sents))
        print("saved: ", path)
        
if __name__ == "__main__":
    load_resource()
    path = "text-restoration-data-new/test"
    files = os.listdir(path)
    files = [os.path.join(path, file) for file in files]

    args = [tuple(files[i:i+1]) for i in range(0, len(files), 1)]
    pool = multiprocessing.Pool(processes=len(args))

    result_list = pool.map(preprocessing_datas, args)