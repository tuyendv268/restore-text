from tqdm import tqdm
import os
import threading
import re

def filter_sent(lines):
    out_sent = []
    for line in lines:
        line = line.replace("\n","").strip()
        if '"' in line or '-' in line or "'" in line or ':' in line or '“' in line or '”' in line or "–" in line:
            continue
        if len(line.split()) <= 6 or "Ảnh" in line:
            continue
        if line.endswith(".") or line.endswith("?") or line.endswith("...") or line.endswith(";") or line.endswith("!"):
            line = line.replace("!", ". ")
            line = line.replace("...", ".")
            out_sent.append(line)
    return out_sent
def save_data(datas, out_path):
    with open(out_path, "w", encoding="utf-8") as tmp:
         for data in datas:
            tmp.write(data+"\n")
    print("\n saved file: ",out_path)

def read_line(f):
    try:
        line = f.readline()
    except:
        print("ignore line")
        return "exception"
    return line
    
def load_data(path):
    f = open(path, "r", encoding="utf-8")
    mark = True
    while mark:
        line = read_line(f)
        if line == "exception":
            continue
        if not line:
            print("Done!")
            mark=False
        yield line

def check_sent(sent):
    line = sent.replace("\n","").strip()
    if '"' in line or ";" in line or '-' in line or "'" in line or ':' in line or '“' in line or '”' in line or "–" in line:
        return False
    if len(line.split()) <= 6 or "Ảnh" in line:
        return False
    if line.endswith(".") or line.endswith("?") or line.endswith("...") or line.endswith(";") or line.endswith("!"):
        return True
    return False

def load_dict(path):
    dictionary  = {}
    with open(path, "r") as tmp:
        dict_ = tmp.readlines()
        for line in dict_:
            tmp_line = line.replace("\n"," ").split("|")
            key = tmp_line[0].strip()
            value = " " +tmp_line[1].strip() + " "
            dictionary[key] = value
    return dictionary

def norm_abbre(text):
    for key, value in dictionary.items():
        text =re.sub(f"[\s+]{key}(?=[\s\.,:;\'\"\!\?])", value, text)
    return text
def remove_bracket(text):
    if '(' not in text and '[' not in text and '{' not in text:
        return text
    text=re.sub("\(.*?\)"," ",text)
    text=re.sub("\[.*?\]"," ",text)
    text=re.sub("\{.*?\}"," ",text)
    return text
def clean_data(inp_path):
    count, idx = 0, 0
    lines = []
    datas = load_data(inp_path)
    dl = tqdm(datas)
    for line in dl:
        line = line.replace("\n","").strip() 
        if not line[0].isalpha():
            continue
        if check_sent(line):
            line = line.replace("! ", ". ")
            line = line.replace("... ", ". ")
            line = line.replace("; ", ". ")
            line = norm_abbre(line)
            line = remove_bracket(line)
            line = line.strip()
            if line.endswith("?") or line.count(",") > 2:
                count += 1
                lines.append(line)
        if count == 100000:
            abs_path = f"data/data_{threading.current_thread().name}_aug_{idx}.txt"
            save_data(lines, abs_path)
            lines = []
            count = 0
            idx += 1
        dl.set_postfix({"step":count, "num_file":idx})
if __name__ == "__main__":
    base_path = "raw"
    path = os.listdir(base_path)
    print(path)
    dictionary = load_dict("resources/dict/abbre.txt")
    for index, file in enumerate(path):
        abs_path = os.path.join(base_path, file)
        print(file)    
        thread = threading.Thread(target=clean_data, name=f't{index}', args=[abs_path])
        thread.start()
    
    