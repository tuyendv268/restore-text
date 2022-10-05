import os
import sys
import re
import string
from tqdm import tqdm
sys.path.append("..")
from multiprocessing import Pool

DATA_PATH = "/home/ngocpt/recoded_data_v1"

def remove_special_characters_v2(input_str):
    return re.sub(r'[^\w\s]', ' ', input_str)

def remove_punct(input_str):
    input_str = ' ' + input_str + ' '
    punct = ', . ?'.split()
    for e in punct:
        e = e.strip()
        input_str = input_str.replace(e, ' ') 
    return input_str

def tokenize(text):
    # text = text.lower().replace('\n', ' ')
    text = remove_punct(text)
    tokens = text.split()
    return tokens

def prepare_input(list_line, range_line):
    input_list = []
    count = 0
    while count < len(list_line):
        input_list.append(' '.join(list_line[count:count + range_line]))
        count += range_line
    return input_list

if __name__ == "__main__": 
    # pool = Pool(processes=10)
    word2count = {} 
    txt_path = '/home/ngocpt/VIN-PROJECT/vnd-nlp-text-processing/src/all_recorded.txt'
    f = open(txt_path)
    lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]
    for line in lines:
        tokens = tokenize(line)
        for token in tokens:
            if token not in word2count.keys():
                word2count[token] = 1
            elif token in word2count.keys():
                word2count[token] += 1

        # count_line = 0
        
        # while True:
        #     texts = prepare_input(lines[count_line:count_line+5000], 500)
        #     input_list = pool.map(tokenize, texts)
        #     input_list = [j for i in input_list for j in i]
            
            # for token in input_list:
            #     if token not in word2count.keys():
            #         word2count[token] = 1
            #     elif token in word2count.keys():
            #         word2count[token] += 1
                    
        #     if count_line > len(lines):
        #         break
        #     count_line += 5000

    f.close()

    sorted_items = dict(sorted(word2count.items(), key=lambda item: item[1], reverse=True))
    outfile = open('vocab_recoded.txt', 'w')
    for key, value in sorted_items.items():
        outfile.write(str(key) + '\t' + str(value) + '\n')