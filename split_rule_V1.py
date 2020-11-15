# import pandas as pd
# import numpy as np
# import re
# import math
# data = pd.read_excel('./row data/200801至202008缺失類型.xlsx',sheet_name = '法規')

def split_rule(df, legal_name_file, split_rule_file):
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

    return sub_data