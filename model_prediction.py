import re
from ckiptagger import WS
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle


def recommend_law(text, model_var_file, ckip_path):
    """
    Load training result and input string to recommend top 10 laws.
    """
    def text_preprocess(raw_text):
        ws = WS(ckip_path)
        rule = re.compile(r'[^a-zA-Z0-9\u4e00-\u9fa5]')
        raw_text = rule.sub(' ', str(raw_text))
        raw_text = re.sub(' +', '', raw_text)
        raw_text = ws([raw_text], sentence_segmentation=True)
        raw_text = [x for l in raw_text for x in l]
        return raw_text

    w2v_vocab, w2v_model, tfidf_feature, tfidf_text_vect, dictionary, df = pickle.load(open(model_var_file, 'rb'))
    text = text_preprocess(text)
    text_vec = np.zeros(100)  # w2v size
    weight_sum = 0
    for word in text:
        if word in w2v_vocab and word in tfidf_feature:
            vec = w2v_model.wv[word]
            tf_idf = dictionary[word] * (text.count(word) / len(text))
            text_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        text_vec /= weight_sum
    # binding all doc vector and add one input vector(seems extend would be better)
    tmp_vec = [*tfidf_text_vect, text_vec]
    new_cos_sim = cosine_similarity(tmp_vec, tmp_vec)
    sim_score = np.sort(new_cos_sim[new_cos_sim.shape[0] - 1])[::-1][1:11]

    tmp_top_10_law = df[['法規名稱', '條', '缺失內容']].iloc[np.argsort(new_cos_sim[new_cos_sim.shape[0] - 1])[::-1][1:11]]
    tmp_top_10_law['similarity_score'] = [round(score * 100, 1) for score in sim_score]
    return tmp_top_10_law