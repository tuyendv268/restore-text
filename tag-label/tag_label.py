from src.utils.utils import *
import sys
sys.path.insert(0, "/home/tuyendv/projects/text-restoration-tag-label/src/resources/norm_text")
from src.processing import run
from tqdm import tqdm
from multiprocessing import Process
import random
import multiprocessing
import os


def tag_token_label(token, next_token):
    if token in '''. , ? ! ... “ ” " # $ % & \' ( ) * + - / : ; < = > @ [ \\ ] ^ _` { | } ~''':
        return None
    tagged = ()
    if next_token not in [",", ".", "?", "!"]:
        if(token[0].isupper()):
            tagged = (token, "upper")
        else:
            tagged = (token, "O")
    else:
        if next_token in ["!"]:
            next_token = "."
        if(token[0].isupper()):
            tagged = (token, "upper"+next_token)
        else:
            tagged = (token, "O"+next_token)
    return [tagged]


def norm_space(text):
    texts = text.split()
    texts = [token for token in texts if len(token) != 0]
    return " ".join(texts)

def save_data(datas, out_path):
    with open(out_path, "w", encoding="utf-8") as tmp:
         for data in datas:
            for token in data:
                tmp.write(token[0] + "\t" + token[1]+"\n")
            tmp.write("\n")
    print("saved file: ",out_path)
    
def tag_label(path):
    outpath = "/home/tuyendv/projects/text-restoration-tag-label/temp/test"
    with open(path, "r", encoding="utf-8") as tmp:
        data = tmp.readlines()
    if "chatbotdatas" in path:
        print("shuffle: ", path)
        random.shuffle(data)
    tagged_sents = []
    pid = os.getpid()
    for sent in tqdm(data, postfix={"pid":pid}):
        sent = sent.replace("\n", "")
        sent = sent.strip()
        # if ":" in sent or sent.count(".") >= 2 or sent.count("?") >= 2 or len(sent) ==0:
        #     continue
        if len(sent) ==0:
            continue
        # sent = run(sent)
        if (sent[0] not in "QWERTYUIOPASDFGHJKLZXCVBNMAĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴ"):
            words = sent.split(" ")
            if words[0].islower() and words[0] not in ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]:
                words[0] = words[0].capitalize()
            sent = " ".join(words).strip()
        if (not sent.endswith(".") and not sent.endswith("?") and not sent.endswith("!")):
            sent = sent + "."
            # continue
        sent = sent.replace(".", " . ").replace("?", " ? ").replace(",", " , ").replace("!", " ! ").strip()
        sent = norm_space(sent)
        sent = sent.split()
        tagged_sent = []
        for index in range(len(sent)-1):
            tmp = tag_token_label(sent[index], sent[index+1])
            if tmp != None:
                tagged_sent+=tmp
        tagged_sents.append(tagged_sent)
        
    outpath_abs = os.path.join(outpath, path.split("/")[-1])
    save_data(tagged_sents, outpath_abs)
    
    return tagged_sents

def do_taglabel(files):
    for file in files:
        tag_label(file)
    

if __name__ == "__main__":
    base_path = "/home/tuyendv/projects/text-restoration-tag-label/text-restoration-data-new/test"
    files = os.listdir(base_path)
    files = [os.path.join(base_path, file) for file in files]
    args = []
    for i in range(0, len(files), 1):
        args.append(files[i:i+1])
        
    pool = multiprocessing.Pool(processes=len(args))

    result_list = pool.map(do_taglabel, args)
    pool.close()
    pool.join()
