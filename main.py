from pathlib import Path
import pandas as pd
# from model_building import data_cleansing, get_punctuation, word_cut, word2vec_model
from model_prediction import recommend_law
import datetime

ckip_path = '/Users/zoe/Desktop/20201107_d4sg_corpus_mapping/data/ckip_model/data'
root_path = Path('/Users/zoe/Desktop/20201107_d4sg_corpus_mapping')
raw_df = pd.read_excel(root_path / 'data/200801至202008缺失類型(法規分段例).xlsx', sheet_name='法規')
parsed_column = '缺失內容'

dict_path = root_path / 'dictionary'
legal_name_file = dict_path / 'name_of_legal.txt'
word_file = dict_path / 'words.txt'
punc_file = dict_path / 'punctuation.pkl'

output_path = root_path / 'output'
step1_file = output_path / 'data_etl_step1.csv'
step2_file = output_path / 'data_etl_step2_noPunc.csv'
step3_file = output_path / 'data_etl_step3_noPuncDict.csv'
w2v_model_file = output_path / 'w2v_var.pkl'

# data_cleansing(raw_df, output_file=step1_file, text_column=parsed_column)
# get_punctuation(step1_file, output_file=punc_file, text_column=parsed_column)
# word_cut(step1_file, step3_file, punc_file, parsed_column, legal_name_file, word_file, ckip_path)
# word2vec_model(step3_file, w2v_model_file)
starttime = datetime.datetime.now()
newtext = '開標時有作拒絕往來廠商調查'
result = recommend_law(newtext, w2v_model_file, ckip_path)
# calculate running time
endtime = datetime.datetime.now()
print("搜尋推薦時間: ", endtime - starttime)
print(result)