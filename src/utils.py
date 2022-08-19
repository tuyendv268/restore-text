from importlib.machinery import SourceFileLoader
from multiprocessing import Pool
import threading
from threading import Thread, current_thread
from multiprocessing import Process
from tqdm import tqdm
import multiprocessing
import torch
from src.resources import hparams
import random
import os
import numpy as np

def extend_label(tokens, labels):
    idx = 0
    new_labels, label_mask = [], []
    for i in range(len(tokens)):
        if tokens[i].startswith("▁") or tokens[i] == "<s>" or tokens[i] == "</s>":
            new_labels.append(labels[idx])
            prev = labels[idx]
            idx += 1
            label_mask.append(1)
        else:
            # if prev.startswith("B"):
            #   new_labels.append(prev.replace("B-","I-"))
            #   prev = prev.replace("B-","I-")
            # else:
            new_labels.append(prev)
            label_mask.append(1)
    return new_labels, label_mask

def cvt_label2ids(label, tag2index):
    labels = [tag2index[ele] for ele in label]
    return labels

def cvt_ids2label(label, index2label):
    labels = [index2label[str(ele)] for ele in label]
    return labels


def convert_data2ids(tokenizer, sent, label, tag2index):
    tokens = tokenizer.tokenize(sent)

    try:
        labels, _ = extend_label(tokens, label)
    except:
        print("Ignore Sent")
        return None, None
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    label_ids = cvt_label2ids(labels, tag2index)
    
    # input_ids = tokens
    # label_ids = labels
    
    return input_ids, label_ids

def prepare_data(input_ids, label_ids, max_sent_length, tokenizer, tag2index):
    max_sent_length += 2
    
    choice = random.choices(hparams.prob_list, weights=hparams.prob_weight, k=1)[0]
    offset = random.randint(2,6)
    
    label_ids = label_ids[label_ids!=tag2index["<pad>"]].tolist()
    input_ids = input_ids[input_ids!=tokenizer.convert_tokens_to_ids("<pad>")].tolist()
    
    label_masks = [1]*len(label_ids)
    input_masks = [1]*len(input_ids)
    
    # print("choice: ", choice)     
    # print("offset:", offset)  
    # print("before: ", input_ids)
    if choice == "cut_both_sides":
        label_ids = [0] + label_ids[offset:-offset] + [2]
        input_ids = [0] + input_ids[offset:-offset] + [2]
        label_masks = [1] + label_masks[offset:-offset]  + [1]
        input_masks = [1] + input_masks[offset:-offset]  + [1]
    elif choice == "cut_left_sides":
        label_ids = [0] + label_ids[offset:] + [2]
        input_ids = [0] + input_ids[offset:] + [2]
        label_masks = [1] + label_masks[offset:] + [1]
        input_masks = [1] + input_masks[offset:] + [1]
    elif choice == "cut_right_sides":
        label_ids = [0] + label_ids[:-offset] + [2]
        input_ids = [0] + input_ids[:-offset] + [2]
        label_masks = [1] + label_masks[:-offset] + [1]
        input_masks = [1] + input_masks[:-offset] + [1]
    elif choice == "stay_same":
        label_ids = [0] + label_ids + [2]
        input_ids = [0] + input_ids + [2]
        label_masks = [1] + label_masks + [1]
        input_masks = [1] + input_masks + [1]
    # print("after: ", input_ids)
    
     
    sent_length = len(input_ids)
    
    input_masks = input_masks + [0]*(max_sent_length-sent_length)
    label_masks = label_masks + [0]*(max_sent_length-sent_length)
    input_ids = input_ids + [1]*(max_sent_length-sent_length)
    label_ids = label_ids + [1]*(max_sent_length-sent_length)

    return input_ids, input_masks, label_ids, label_masks

def prepare_data_for_infer(tokenizer, sent):
    tokens = tokenizer.tokenize(sent)

    tokens = ["<s>"] + tokens + ["</s>"]
    input_masks = [1]*len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    return input_ids, input_masks

def remove_padding(labels, masks):
    label = [sent[0:mask.sum(dim=0)] for sent, mask in zip(labels, masks)]
    return label
    
def convert_id2label(input_ids,index2label):
    output = [[index2label[str(token)] for token in inp] for inp in input_ids]
    return output

def ignore_label(predicts, labels, ignore_label):
    predicts = [[pred for pred, lbl in zip(sent_pred, sent_label) if lbl not in ignore_label] for sent_pred, sent_label in zip(predicts, labels)]
    labels = [[token for token in sent_label if token not in ignore_label] for sent_label in labels]

    return predicts, labels

def merge_sent(input_ids, label_ids):
    n = len(input_ids)
    
    new_datas, new_labels = [], []
    datas_temp, label_temp = [], []
    current_len, i = 0, 0
    while i < n:
        idx = i
        
        while(idx < n):
            if current_len + len(input_ids[idx]) > hparams.max_sent_length:
                current_len = 0
                datas_temp += [-1]*(hparams.max_sent_length-len(datas_temp))
                label_temp += [-1]*(hparams.max_sent_length-len(label_temp))
                
                new_datas.append(datas_temp)
                new_labels.append(label_temp)
                                
                datas_temp, label_temp = [], []
                idx -= 1
            else:
                # print(str(len(input_ids)) + " - "+str(idx))
                current_len +=  len(label_ids[idx])
                datas_temp += input_ids[idx]
                label_temp += label_ids[idx]  
            idx += 1
        i = idx + 1
    return new_datas, new_labels

def load_data_parallel(path, tokenizer, tag2index, max_sent_length):
    files = os.listdir(path)
    files = [os.path.join(path, file) for file in files]
    
    
    # args = [(folder, tag2index, max_sent_length) for folder in folders]
    print("total: ", len(files))
    num_cores = hparams.num_cores
    print("num_cores: ", num_cores)
    files = transform_list(files, num_cores)    
    
    
    pool = Pool(processes=num_cores)
    
    # thread_lists = []
    # for index, file in enumerate(files):
        # thread_tmp = Thread(name=f"thread_{index}",target=load_data,args=(file,tag2index, max_sent_length))
    args = [(file,tag2index, max_sent_length) for file in files]
    res = pool.starmap(load_data, args)
        # thread_lists.append(thread_tmp)
        # thread_tmp.start()
    pool.close()
    pool.join()
    print("finished")
    
    res = (i for i in res)
    input_ids, label_ids = [], []
    for input_id, label_id in res:
        input_ids += input_id
        label_ids += label_id        
    return input_ids, label_ids

def transform_list(files, num_cores):
    n = int(len(files)/num_cores) + 1
    threads = []
    for i in range(num_cores):
        start = i * n
        end = (i + 1) * n
        threads.append(files[start:end])
    return threads
    

# def load_data_parallel(path, threadLock, tag2index, max_sent_length):
#     global datas_global
#     global labels_global
    
#     datas_global, labels_global = [], []
#     files = os.listdir(path)
#     files = [os.path.join(path, file) for file in files]
    
    
#     # args = [(folder, tag2index, max_sent_length) for folder in folders]
#     print("total: ", len(files))
#     num_cores = hparams.num_cores
#     print("num_cores: ", num_cores)
#     files = transform_list(files, num_cores)    
    
#     thread_lists = []
#     for index, file in enumerate(files):
#         thread_tmp = Thread(name=f"thread_{index}",target=load_data,args=(file, threadLock ,tag2index, max_sent_length))
#         thread_lists.append(thread_tmp)
#         # thread_tmp.start()
        
#     for process_tmp in thread_lists:
#         process_tmp.start()
    
#     for process_tmp in thread_lists:
#         process_tmp.join()
        
#     print("finished")
#     datas_global = (i for i in datas_global)
#     input_ids = [list(sent) for sent in datas_global]
#     datas_global = None
    
#     labels_global = (i for i in labels_global)
#     label_ids = [list(sent) for sent in labels_global]
#     labels_global = None
        
#     return input_ids, label_ids

def load_data(files, tag2index, max_sent_length):
    tokenizer = SourceFileLoader("envibert.tokenizer", 
                    os.path.join(hparams.pretrained_envibert,'envibert_tokenizer.py'))\
                        .load_module().RobertaTokenizer(hparams.pretrained_envibert)
    datas_global = []
    labels_global = []
    for input_ids, label_ids in load_folder(files, tokenizer, tag2index, max_sent_length):
        # threadLock.acquire()
        for input_id, label_id in zip(input_ids, label_ids):
            # print(input_id)
            # print("shape: ", len(input_id))
            datas_global.append(input_id)
            labels_global.append(label_id)
    return (datas_global, labels_global)
        # threadLock.release() 
                        
def load_file(path, tokenizer, tag2index, max_sent_length):
    input_ids, label_ids = [], []
    # count = 0
    
    with open(path, "r",encoding="utf-8") as tmp:
        lines = tmp.readlines()
        datas_temp, labels_temp = "", []
        for line in lines:
            temp = line.replace("\n","").split("\t")
            if(line == "\n"):
                # count += 1
                input_id, label_id = convert_data2ids(tokenizer, datas_temp, labels_temp, tag2index)
                
                if input_id == None or label_id == None:
                    datas_temp, labels_temp = "", []
                    print("ignore")
                    continue
                
                len_tmp = len(input_id)
                if len_tmp > max_sent_length:
                    input_id = input_id[0:max_sent_length]
                    label_id = label_id[0:max_sent_length]
                    continue
                # else:
                #     input_id = input_id + [-1]*(max_sent_length - len_tmp)
                #     label_id = label_id + [-1]*(max_sent_length - len_tmp)
                    
                # input_ids.append(input_id)
                # label_ids.append(label_id)
                
                # if count == 50:
                yield input_id, label_id
                    # input_ids, label_ids = [], []
                    # count = 0
                    
                datas_temp, labels_temp = "", []
            else:
                datas_temp += temp[0].lower() + " "
                if "?" in temp[1]:
                    temp[1] = temp[1].replace("?",".")
                labels_temp.append(temp[1])
    # print(label_ids)
    # return input_ids, label_ids

def load_folder(files, tokenizer, tag2index, max_sent_length):
    thread = current_thread()
    name=thread.name
    for file in tqdm(files, total=len(files), postfix={"process":name}):        
        input_ids, label_ids =[], []
        input_ids_tmp, label_ids_tmp =[], []
        current_len, count_sample = 0, 0
        
        for input_id, label_id in load_file(path=file, tokenizer=tokenizer, tag2index=tag2index, max_sent_length=max_sent_length):
            length = len(input_id)
            
            if current_len + length< hparams.max_sent_length:
                input_ids_tmp += input_id;label_ids_tmp += label_id
                current_len += length
            # input_ids, label_ids = merge_sent(input_ids, label_ids)
            else:
                # if len(input_ids_tmp) != len(label_ids_tmp):
                #     print("Error")
                input_ids_tmp = input_ids_tmp + [tokenizer.convert_tokens_to_ids("<pad>")] * (max_sent_length-len(input_ids_tmp))
                label_ids_tmp = label_ids_tmp + [tag2index["<pad>"]] * (max_sent_length-len(label_ids_tmp))
                
                input_ids.append(input_ids_tmp); label_ids.append(label_ids_tmp)
                count_sample += 1

                if count_sample == hparams.yield_every:
                    count_sample = 0
                    yield input_ids, label_ids
                    input_ids, label_ids =[], []
                current_len = length
                input_ids_tmp = input_id
                label_ids_tmp = label_id

def join(sp_token, sp_tag):
    join_token, join_tag = [], []
    tmp_token, tmp_tag = [], []
    for token, tag in zip(sp_token, sp_tag):
        token = str(token)
        if token.startswith('▁') or token == '<s>' or token == '</s>':
            if len(tmp_token) > 0:
                join_token.append(''.join(tmp_token).replace('▁', ''))
                join_tag.append(tmp_tag[0])
            tmp_token = [token]
            tmp_tag = [tag]
        else:
            tmp_token.append(token)
            tmp_tag.append(tag)

    if len(tmp_token) > 0:
        join_token.append(''.join(tmp_token).replace('▁', ''))
        join_tag.append(tmp_tag[0])

    return join_token[1:-1], join_tag[1:-1]

def joins(sp_tokens, sp_tags):
    join_tokens, join_tags = [], []
    for sp_token, sp_tag in zip(sp_tokens, sp_tags):
        join_token, join_tag = join(sp_token, sp_tag)
        join_tokens.append(join_token)
        join_tags.append(join_tag)
    return join_tokens, join_tags