#!/usr/bin/env python
# coding: utf-8

import re
import pickle
from string import punctuation
import datetime

from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
from ckiptagger import construct_dictionary, WS
from sklearn.feature_extraction.text import TfidfVectorizer


def exclude_rule(df, output_file, legal_name_file, split_rule_file):
    # input some dictionaries
    # rule_table = pd.read_csv('./dictionary/rule.csv',encoding = 'big5')

    rule_names = []
    with open(legal_name_file, 'r', encoding='big5') as file :
        for i in file.readlines():
            rule_names.append(i.strip())

    add_rule_names = []
    with open(split_rule_file, 'r', encoding='utf8') as file :
        for i in file.readlines():
            add_rule_names.append(i.strip())
    rule_names.extend(add_rule_names)

    sub_data = df
    input_re_string_1 = '|'.join(rule_names)
    sub_data.columns = [i.split('\n')[0] for i in sub_data.columns.tolist()]
    # re_exp_1 = '(?:{})[^：]*[：|:|略以|載有|規定]「?[^「|」]*[」|。]'.format(input_re_string_1)
    # 較嚴肅
    # re_exp_1 = '(?:{})[^：|。|，]*[：|:|，略以|載有|。|規定](?:「[^「|」|]*|[^「|」|。]*)[」|。]'.format(input_re_string_1)
    # 較寬鬆
    re_exp_1 = '(?:{})[^：|。|，]*[：|:|，略以|載有|。|規定](?:「[^「|」]*)[」|。]'.format(input_re_string_1)
    sub_data['法令依據_cloud'] = sub_data['缺失內容'].str.findall(re_exp_1)
    sub_data['事實&改進建議'] = sub_data['缺失內容'].str.replace(re_exp_1, '')

    input_re_string_2 = '|'.join(rule_names)
    re_exp_2 = input_re_string_2
    sub_data['法令_cloud'] = sub_data['缺失內容'].str.findall(re_exp_2)

    news_list = []
    for i in sub_data['法令_cloud']:
        if type(i) != list :
            news_list.append([])
        else:
            news_list.append(list(set(i)))
    sub_data['法令_cloud'] = news_list
    sub_data['缺失內容'] = sub_data['事實&改進建議']
    sub_data.to_excel(output_file, index=False)
    print(output_file, ' exported.')


def data_cleansing(df, output_file, text_column):
    """
    cleansing DataFrame then export csv result.
    """
    # trans columns to chinese
    df.columns = [x.split('\n')[0] for x in df.columns]
    # filter NA
    df = df[df[text_column].notna()]  # 缺失內容非空 (22483, 9)
    df = df[df[text_column].apply(lambda x: pd.to_numeric(x, errors='coerce')).isna()]  # 缺失內容非單一數字 (22481, 9)
    df = df[df['條'].notna()]  # 條非空 (15198, 9)
    replace_dict = {
        '零': '0'
        , '壹': '1'
        , '貳': '2'
        , '參': '3'
        , '肆': '4'
        , '伍': '5'
        , '陸': '6'
        , '柒': '7'
        , '捌': '8'
        , '玖': '9'
        , '一': '1'
        , '二': '2'
        , '三': '3'
        , '四': '4'
        , '五': '5'
        , '六': '6'
        , '七': '7'
        , '八': '8'
        , '九': '9'
        , '拾': '10'
        , '佰': '100'
        , '十': '10'
        , '兩': '2'
        , '倆': '2'
    }
    df.replace({text_column: replace_dict}, regex=True, inplace=True)
    df.to_csv(output_file, index=False)
    print(output_file, ' exported.')


def get_punctuation(file_path, output_file, text_column):
    df = pd.read_csv(file_path)

    punc = list()
    for i, s in enumerate(df[text_column]):
        s = re.findall(r'[^\u4e00-\u9fff]+', s)
        s = ''.join(s)
        s = re.findall(r'[^\w]+', s)
        s = ''.join(s)
        punc = punc + list(s)

    punc = set(punc)
    punc = punc.union(punctuation).union({'\n', '\t', '【', '】', '「', '」', '.', '。'})
    pickle.dump(punc, open(output_file, 'wb'))
    print(output_file, ' exported.')


def word_cut(file_path, output_file, punc_pkl, text_column, legal_name_file, word_file, ckip_path):
    ws = WS(ckip_path)
    with open(legal_name_file, 'r', encoding='big5') as k1, open(word_file, 'r', encoding='big5') as k2:
        k = k1.read().split('\n') + k2.read().split('\n')
        word_to_weight = dict([(_, 1) for _ in k])
    dictionary = construct_dictionary(word_to_weight)

    df = pd.read_csv(file_path)
    punc = pickle.load(open(punc_pkl, 'rb'))
    word_s = ws(df[text_column],
                sentence_segmentation=True, segment_delimiter_set=punc, recommend_dictionary=dictionary)
    # filter and output
    word_s1 = [[_ for _ in w if _ not in punc] for w in word_s]
    df['token'] = ['@'.join(_) for _ in word_s1]
    df.to_csv(output_file, index=False)
    print(output_file, ' exported.')


def word2vec_model(file_path, output_file):
    df = pd.read_csv(file_path)
    df = df[df['token'].notna()]
    # Replace '@' with ' ' in original dataframe
    df.token = df.token.apply(lambda text: text.replace('@', ' '))

    tfidf_ml = TfidfVectorizer()
    tfidf_ml.fit(df.token)

    # TF-IDF Dicitonary
    dictionary = dict(zip(tfidf_ml.get_feature_names(), list(tfidf_ml.idf_)))

    # feature name
    tfidf_feature = tfidf_ml.get_feature_names()

    w2v_model = Word2Vec(df.token.apply(lambda text: text.split()))
    w2v_vocab = list(w2v_model.wv.vocab)
    print(w2v_model)

    starttime = datetime.datetime.now()

    # TF-IDF weighted Word2Vec
    tfidf_text_vect = []  # tfidf-w2v is stored in this list
    row = 0

    for text in df.token.apply(lambda text: text.split()):
        text_vect = np.zeros(100)
        weight_sum = 0
        for word in text:
            if word in w2v_vocab and word in tfidf_feature:
                vec = w2v_model.wv[word]
                tf_idf = dictionary[word] * (text.count(word) / len(text))
                text_vect += (vec * tf_idf)
                weight_sum += tf_idf
        if weight_sum != 0:
            text_vect /= weight_sum
        tfidf_text_vect.append(text_vect)
        row += 1

    # calculate running time
    endtime = datetime.datetime.now()
    print("建立模型時間: ", endtime - starttime)
    model_var = [w2v_vocab, w2v_model, tfidf_feature, tfidf_text_vect, dictionary, df]
    pickle.dump(model_var, open(output_file, 'wb'))
    print(output_file, ' exported.')
