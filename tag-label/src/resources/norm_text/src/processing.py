import os
import re
import sys
sys.path.append("..")
import pandas as pd
from multiprocessing import Pool
from nltk import sent_tokenize
from configs.abbre import ABBRE
from base.normalizer import TextNormalizer

def normalize(text):
    try:
        normalizer = TextNormalizer()
        text = normalizer.remove_urls(text)
        # print(text)
        # text = normalizer.normalize_urls(text)
        # print(text)
        text = normalizer.remove_emoji(text)
        # text = normalizer.remove_emoticons(text)
        # print(text)
        text = normalizer.remove_special_characters_v1(text)
        # print(text)
        # text = normalizer.norm_abbre(text, ABBRE)
        # print(text)
        text = normalizer.normalize_number_plate(text)
        # text = normalizer.norm_punct(text)
        # print(text)
        text = normalizer.norm_tag_measure(text)
        # print(text)
        text = normalizer.normalize_rate(text)
        # print(text)
        text = normalizer.norm_adress(text)
        # print(text)
        text = normalizer.norm_tag_fraction(text)
        # print(text)
        text = normalizer.norm_multiply_number(text)
        # print(text)
        text = normalizer.normalize_sport_score(text)
        # print(text)
        text = normalizer.normalize_date_range(text)
        # print(text)
        text = normalizer.normalize_date(text)
        # print(text)
        text = normalizer.normalize_time(text)
        # print(text)
        text = normalizer.normalize_number_range(text)
        # print(text)
        text = normalizer.norm_id_digit(text)
        # print(text)
        text = normalizer.norm_soccer(text)
        # print(text)
        text = normalizer.normalize_negative_number(text) 
        # print(text)
        text = normalizer.norm_tag_roman_num(text)
        # print(text)
        # text = normalizer.normalize_AZ09(text)
        # print(text)
        text = normalizer.norm_math_characters(text)
        # print(text)
        text = normalizer.norm_tag_verbatim(text)
        # print(text)
        text = normalizer.normalize_number(text)
        # print(text)
        text = normalizer.normalize_phone_number(text)
        # print(text)
        # text = normalizer.norm_account_name(text)
        # text = normalizer.normalize_letters(text)
        # print(text)
        text = normalizer.norm_vnmese_accent(text)
        # print(text)
        # print(text)
        text = text.replace('/', ' trên ')
        # print(text)
        # text = normalizer.remove_special_characters_v2(text)
        # print(text)
        text = normalizer.norm_tag_roman_num(text)
        # print(text)
        text = normalizer.remove_multi_space(text)
        # print(text)
        text = normalizer.normalize_number(text)
        # print(text)
        # text = normalizer.lowercase(text)
        # text = text.replace('. ', ' . ')
        
        # text = text.replace('? ', ' ? ')
        # text = text.replace(', ', ' , ')
        # text = normalizer.norm_duplicate_word(text)
        # text = text.replace('. . .', '...')
        # print(text)
        return text
    except:
        print("ignore sent")
        return None

def run(text):
    # norm_text = []
    # for sentence in sent_tokenize(text):
    text = normalize(text)
    #     if len(sentence.split()) <= 3:
    #         continue
    #     norm_text.append(sentence)
    # text = ' '.join(norm_text)
    if text == None:
        return None
    text = re.sub(' +', ' ', text)
    while True:
        if len(text) == 0 or text[0] != '.':
            break
        else:
            text = text[1:]
    return text
