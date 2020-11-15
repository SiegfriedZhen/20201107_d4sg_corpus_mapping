from pathlib import Path
import pandas as pd
from model_building import data_cleansing, get_punctuation, word_cut, word2vec_model, exclude_rule
from model_prediction import recommend_law
import datetime

# word cut package data path and exec path
root_path = Path('/Users/zoe/Documents/GitHub/20201107_d4sg_corpus_mapping')
ckip_path = root_path / 'data/ckip_model/data'
parsed_column = '缺失內容'
# dictionary
dict_path = root_path / 'dictionary'
legal_name_file = dict_path / 'name_of_legal.txt'
word_file = dict_path / 'oth_words.txt'
punc_file = dict_path / 'punctuation.pkl'
split_rule_kw_file = dict_path / 'split_rule_words.txt'
# output file name
output_path = root_path / 'output/include_rule_ver'
# without rule exclusion
step1_file = output_path / 'data_etl_step1_exclude_law.xlsx'  # output as excel to avoid comma parse error
step2_file = output_path / 'data_etl_step2.csv'
step3_file = output_path / 'data_etl_step3_noPuncDict.csv'
w2v_model_file = output_path / 'w2v_var.pkl'
# ignore this no dict version
# output_path / 'data_etl_step2_noPunc.csv'

# # load raw data
raw_df = pd.read_excel(root_path / 'data/200801至202008缺失類型(法規分段例).xlsx', sheet_name='法規')
# # exclude law
# exclude_rule(raw_df, step1_file, legal_name_file, split_rule_kw_file)
# raw_df = pd.read_excel(step1_file)
# # preprocessing
data_cleansing(raw_df, output_file=step2_file, text_column=parsed_column)
get_punctuation(step2_file, output_file=punc_file, text_column=parsed_column)
word_cut(step2_file, step3_file, punc_file, parsed_column, legal_name_file, word_file, ckip_path)
# model building
word2vec_model(step3_file, w2v_model_file)
# recommendation
starttime = datetime.datetime.now()
newtext = '開標時有作拒絕往來廠商調查'
# if word segment replace to jieba maybe exe size would smaller
result = recommend_law(newtext, w2v_model_file, ckip_path)
# calculate running time
endtime = datetime.datetime.now()
print("搜尋推薦時間: ", endtime - starttime)
print(result)
