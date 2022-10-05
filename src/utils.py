from importlib.machinery import SourceFileLoader
from multiprocessing import Pool
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from tqdm import tqdm
from src.resources import hparams
import random
import os

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
    
    label_ids = label_ids[label_ids!=1].tolist()
    input_ids = input_ids[input_ids!=1].tolist()
    
    label_masks = [1]*len(label_ids)
    input_masks = [1]*len(input_ids)
    
    if len(input_ids) < 3*offset :
        choice = "stay_same"
        
    start = offset
    end = -offset
    
    # while(not tokenizer.convert_ids_to_tokens(input_ids[start]).startswith("▁")):
    #     start += 1
    #     if start >= len(input_ids):
    #         print("out of length")
    #         start = offset
    #         break
    # while(not tokenizer.convert_ids_to_tokens(input_ids[end-1]).startswith("▁")):
    #     end -= 1
    #     if end >= -len(input_ids)+1:
    #         print("out of length")
    #         end = -offset
    #         break
    # end -= 1
    
    if choice == "cut_both_sides":
        label_ids = [0] + label_ids[start:end] + [2]
        input_ids = [0] + input_ids[start:end] + [2]
        label_masks = [1] + label_masks[start:end]  + [1]
        input_masks = [1] + input_masks[start:end]  + [1]
    elif choice == "cut_left_sides":
        label_ids = [0] + label_ids[start:] + [2]
        input_ids = [0] + input_ids[start:] + [2]
        label_masks = [1] + label_masks[start:] + [1]
        input_masks = [1] + input_masks[start:] + [1]
    elif choice == "cut_right_sides":
        label_ids = [0] + label_ids[:end] + [2]
        input_ids = [0] + input_ids[:end] + [2]
        label_masks = [1] + label_masks[:end] + [1]
        input_masks = [1] + input_masks[:end] + [1]
    elif choice == "stay_same":
        label_ids = [0] + label_ids + [2]
        input_ids = [0] + input_ids + [2]
        label_masks = [1] + label_masks + [1]
        input_masks = [1] + input_masks + [1]
     
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

def load_data_parallel(path, tag2index, max_sent_length):
    files = [os.path.join(path, file) for file in os.listdir(path)]

    print("total: ", len(files))
    num_cores = hparams.num_cores
    print("num_cores: ", num_cores)
    files = transform_list(files, num_cores)    
    
    pool = Pool(processes=num_cores)
    
    args = [(file, tag2index, max_sent_length) for file in files]
    res = pool.starmap(load_data, args)

    pool.close(); pool.join()
    print("finished")
    
    res = (i for i in res)
    input_ids, label_ids = [], []
    for input_id, label_id in res:
        input_ids += input_id
        label_ids += label_id 
    res = None       
    return input_ids, label_ids

def transform_list(files, num_cores):
    n = int(len(files)/num_cores) + 1
    threads = []
    for i in range(num_cores):
        start = i * n
        end = (i + 1) * n
        threads.append(files[start:end])
    return threads

def load_data(files, tag2index, max_sent_length):
    if hparams.toknizer_path == None:
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    else:
        tokenizer = SourceFileLoader("envibert.tokenizer", 
                    os.path.join(hparams.toknizer_path,'envibert_tokenizer.py'))\
                        .load_module().RobertaTokenizer(hparams.toknizer_path)
    datas_global, labels_global = [], []
    
    for input_ids, label_ids in load_folder(files, tokenizer, tag2index, max_sent_length):
        for input_id, label_id in zip(input_ids, label_ids):
            datas_global.append(input_id)
            labels_global.append(label_id)
            
            
    return (datas_global, labels_global)
                        
def load_file(path, tokenizer, tag2index, max_sent_length):
    with open(path, "r",encoding="utf-8") as tmp:
        lines = tmp.readlines()
        datas_temp, labels_temp = "", []
        for line in lines:
            temp = line.replace("\n","").split("\t")
            if(line == "\n"):
                input_id, label_id = convert_data2ids(tokenizer, datas_temp, labels_temp, tag2index)
                
                if input_id == None or label_id == None:
                    datas_temp, labels_temp = "", []
                    print("ignore")
                    continue
                
                len_tmp = len(input_id)
                if len_tmp > max_sent_length:
                    input_id = input_id[0:max_sent_length]
                    label_id = label_id[0:max_sent_length]
                    datas_temp, labels_temp = "", []
                    continue

                yield input_id, label_id
                    
                datas_temp, labels_temp = "", []
            else:
                datas_temp += temp[0].lower() + " "
                if "?" in temp[1]:
                    temp[1] = temp[1].replace("?",".")
                labels_temp.append(temp[1])

def load_folder(files, tokenizer, tag2index, max_sent_length):
    for file in tqdm(files, total=len(files)):        
        input_ids, label_ids =[], []
        input_ids_tmp, label_ids_tmp =[], []
        current_len, count_sample = 0, 0
        for input_id, label_id in load_file(path=file, tokenizer=tokenizer, tag2index=tag2index, max_sent_length=max_sent_length):
            length = len(input_id)
            
            if current_len + length< max_sent_length:
                input_ids_tmp += input_id;label_ids_tmp += label_id
                current_len += length
            else:
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

def remove_punct(text):
    punct = '''!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'''
    text = [char for char in text if char not in punct]
    text = "".join(text)
    text = text.lower()
    return text

def restore(tokens, labels):
    res = ''
    index = 0
    while index < len(tokens):
        if labels[index].endswith('upper'):
            res += tokens[index].capitalize() + " "
        elif labels[index] == "O":
            res += tokens[index] + " "
        elif 'upper' in labels[index]:
            res += tokens[index].capitalize() + labels[index].replace("upper","")+" "
        elif 'O' in labels[index]:
            res += tokens[index] + labels[index].replace("O","")+" "
        index += 1
    return res