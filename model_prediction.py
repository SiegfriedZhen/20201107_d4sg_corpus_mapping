import re
from ckiptagger import WS
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from numpy import linalg as LA
from numpy import dot
import pickle
import pandas as pd


def recommend_law(text, model_var_file, word_dict_file, ckip_path):
    """
    Load training result and input string to recommend top 10 laws.
    """
    def text_preprocess(raw_text):
        word_dict = pickle.load(open(word_dict_file, 'rb'))
        ws = WS(ckip_path)
        rule = re.compile(r'[^a-zA-Z0-9\u4e00-\u9fa5]')
        raw_text = rule.sub(' ', str(raw_text))
        raw_text = re.sub(' +', '', raw_text)
        raw_text = ws([raw_text], sentence_segmentation=True, recommend_dictionary=word_dict)
        raw_text = [x for l in raw_text for x in l]
        return raw_text

    embeddings, tfidf_feature, tfidf_text_vect, dictionary, df = pickle.load(open(model_var_file, 'rb'))
    # calculation part
    text = text_preprocess(text)
    text_vect = np.zeros(400) # w2v size
    weight_sum = 0
    for word in text:
        if word in embeddings.keys() and word in tfidf_feature:
            vec = embeddings[word]
            tf_idf = dictionary[word]*(text.count(word)/len(text))
            text_vect += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        text_vect /= weight_sum
    # binding all doc vector and add one input vector(seems extend would be better)
    target_vec = text_vect
    all_vector = np.array(tfidf_text_vect)
    if LA.norm(target_vec) == 0:
        sim = dot(target_vec, all_vector.T) / (1 * LA.norm(all_vector, axis=1))
    else:
        sim = dot(target_vec, all_vector.T) / (LA.norm(target_vec) * LA.norm(all_vector, axis=1))

    rank = np.sort(sim)[::-1]
    nan_cnt = np.isnan(rank).sum()
    sim_score = rank[nan_cnt:10 + nan_cnt]
    tmp_top_10_law = df[['法規名稱', '條', '事實&改進建議']].iloc[np.argsort(sim)[::-1][nan_cnt:10 + nan_cnt]]
    tmp_top_10_law['similarity_score'] = [round(score * 100, 1) for score in sim_score]
    print('推薦結果如下：')
    return tmp_top_10_law


def evaluation(file_path, output_file, model_var_file, word_dict_file, ckip_path):
    print(file_path)
    df = pd.read_excel(file_path)
    df = df[df['法規名稱'] != '行政疏失']
    df['條'] = df['條'].astype(int).apply(str)
    output_df = pd.DataFrame()

    for evl_type in ['原始版', '定稿版']:
        print(evl_type)
        evl_type_df = pd.DataFrame()
        score = 0
        for row in df[['法規名稱', '條', evl_type]].itertuples():
            print(row[0])
            predict_df = recommend_law(row[3], model_var_file, word_dict_file, ckip_path)
            predict_df['prediction_result'] = ((predict_df['法規名稱'] == row[1]) & (predict_df['條'] == row[2]))
            predict_df['evl_index'] = row.Index
            evl_type_df = evl_type_df.append(predict_df)
            if sum(predict_df['prediction_result']) > 0:
                print('correct!')
                score = score + 1
        evl_type_df['evl_type'] = evl_type
        output_df = output_df.append(evl_type_df)
        print(f'Score: {score}, ', round(score/46 * 100, 1), "分")
    output_df.to_excel(output_file, index=False)
    print(output_file, ' exported.')

