import os
import regex
import sys
sys.path.insert(0, "src/resources/norm_text")
from src.processing import run

def save_data(datas, path):
    with open(path, "w", encoding="utf-8") as tmp:
        tmp.write("\n".join(datas))
    print("saved: ", path)
    
def is_ignore(sent):
    if ("..." in sent):
        return True
    if (sent[0] not in "QWERTYUIOPASDFGHJKLZXCVBNMAĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴ"):
        return True
    if (not sent.endswith(".") and not sent.endswith("?") and not sent.endswith("!")):
        return True
    
def load_text(path):
    path = "/home/tuyendv/projects/text-restoration-tag-label/data_raws/data.txt"
    with open(path, "r", encoding="utf-8") as tmp:
        data = tmp.read()
    pattern = "\n\d+\n\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}\n"
    datas = regex.split(pattern, data)
    clean_data = []
    for line in datas[1:]:
        line = line.replace("<i>", " ")
        line = line.replace("</i>", " ")
        tmps = line.split("\n")        
        clean_data+=tmps
    clean_data = [sent.strip() for sent in clean_data if len(sent) != 0]
    
    return clean_data

def merge_sent(lines):
    num_lines = len(lines)

    datas = []
    index = 0
    while(index < num_lines-1):
        if lines[index].endswith("...") or lines[index].endswith(",") or lines[index][-1] in 'qwertyuiopasdfghjklzxcvbnmaăâáắấàằầảẳẩãẵẫạặậđeêéếèềẻểẽễẹệiíìỉĩịoôơóốớòồờỏổởõỗỡọộợuưúứùừủửũữụựyýỳỷỹỵ':
            tmp_sent = [lines[index], ]
            tmp_index = index+1
            while(lines[tmp_index][0].islower()):
                tmp_sent.append(lines[tmp_index])
                tmp_index += 1
            index = tmp_index-1
            datas.append(" ".join(tmp_sent))
        else:
            datas.append(lines[index])
        index += 1
    datas = [data.replace("-", " ").strip() for data in datas]
    return datas

def load_file(path):
    contents = load_text(path)
    datas = merge_sent(contents)
    datas = [data for data in datas if not is_ignore(data)]
    normed_datas = []
    for data in datas:
        normed_data = run(data).strip()
        normed_datas.append(normed_data)
    
    return normed_datas

path = "/home/tuyendv/projects/text-restoration-tag-label/data_raws/data.txt"
datas = load_file(path)