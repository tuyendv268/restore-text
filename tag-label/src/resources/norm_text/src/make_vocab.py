import os
import sys
import re
import pandas as pd
from tqdm import tqdm
sys.path.append("..")
from base.normalizer import TextNormalizer
from collections import Counter
from vncorenlp import VnCoreNLP
from configs.path import VIETNAMESE_STOPWORDS

NEWS_PATH = '/home/ngocpt/VIN-PROJECT/reviewed_data'
wordlist18k = "/home/ngocpt/VIN-PROJECT/wordlist18k.txt"
# NEWS_PATH = '/home/ngocpt/VIN-PROJECT/vnd-nlp-text-crawler/test_data'

normalizer = TextNormalizer()
# annotator = VnCoreNLP(address="http://127.0.0.1", port=9000) 
# stopwords = [line.rstrip("\n") for line in open(VIETNAMESE_STOPWORDS)]

# def vocabulary():
#     # word2count = {}
#     for filename in tqdm(os.listdir(NEWS_PATH)):
#         f =  open(os.path.join(NEWS_PATH, filename), 'r')
#         texts = f.readlines()
#         f.close()
#         count_line = 0
#         print("Processing file %s" %(filename))
#         while True:
#             try:
#                 # tokenizers = []
#                 phrases = []
#                 text = ' '.join(texts[count_line:count_line+1])
#                 text = normalizer.remove_urls(text)
#                 # text = normalizer.remove_emoji(text)
#                 # text = normalizer.remove_emoticons(text)
#                 text = normalizer.remove_html(text)
#                 # text = normalizer.remove_punctuations(text)
#                 text = normalizer.remove_number(text)
#                 # text_tokenizer = annotator.tokenize(text)
#                 list_phrase = [phrase.strip() for phrase in text.split(',')]
#                 for phrase in list_phrase:
#                     if len(phrase.split()) <= 5:
#                         phrases.append(phrase)

#                 # for i in range(len(text_tokenizer)):
#                 #     tokenizers.extend(text_tokenizer[i])
#                 # tokenizers = [token.lower() for token in tokenizers]
#                 phrases = [phrase.lower() for phrase in phrases]
#                 for phrase in phrases:
#                     if phrase not in phrase2count.keys() and phrase not in stopwords:
#                         phrase2count[phrase] = 1
#                     elif token in word2count.keys() and phrase not in stopwords:
#                         phrase2count[phrase] += 1
#                 # for token in tokenizers:
#                 #     if token not in word2count.keys() and len(token) > 1 and token not in stopwords:
#                 #         word2count[token] = 1
#                 #     elif token in word2count.keys() and len(token) > 1 and token not in stopwords:
#                 #         word2count[token] += 1
#             except:
#                 pass
#             count_line += 1
#             # print(count_line, '/', len(texts))
#             if count_line  > len(texts):
#                 break
#     sorted_items = dict(sorted(phrase2count.items(), key=lambda item: item[1], reverse=True))
#     f = open('vocab.txt', 'w')
#     for i in sorted_items:
#         f.write(str(i) + '\n')

def vocabulary():
    word2count = {}
    f = open(wordlist18k)
    wordlist = f.readlines()
    wordlist = [word.rstrip('\n').strip() for word in wordlist]
    for word in wordlist:
        word2count[word] = 0

    for filename in tqdm(os.listdir(NEWS_PATH)):
        print(os.path.join(NEWS_PATH, filename))
        df = pd.read_csv(os.path.join(NEWS_PATH, filename))
        df = df[df['reviewed'].notna()]
        comments = df["reviewed"]
        print("Processing file %s" %(filename))
        count = 0
        while count < len(comments):
            tokenizers = []
            text = comments[count]
            text = normalizer.remove_urls(text)
            text = normalizer.remove_emoji(text)
            # text = normalizer.remove_emoticons(text)
            text = normalizer.remove_html(text)
            text = normalizer.norm_punct(text)
            text = normalizer.remove_number(text)
            # text_tokenizer = annotator.tokenize(text)
            text = re.sub(' +', ' ', text)
            tokens = [token.lower() for token in text.split()]
            for token in tokens:
                if token in list(word2count.keys()):
                    word2count[token] += 1
            count += 1
            # print(count_line, '/', len(texts))
    # sorted_items = dict(sorted(word2count.items(), key=lambda item: item[1], reverse=True))
    f = open('word_statistics.txt', 'w')
    for key, value in word2count.items():
        f.write(str(key) + ' ' + str(value) + '\n')

if __name__ == "__main__":
    vocabulary()
        