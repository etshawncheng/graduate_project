import pandas as pd
import numpy as np
from re import sub
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from jieba.analyse import extract_tags, textrank
from ast import literal_eval
# function to get keywords from a text


def addkeywords():
    def get_keywords(plot: list([str])) -> list([str]):
        # extract keywords from text
        # get dictionary with keywords and scores
        # e=extract_tags(alldata["description"][0], withWeight=True)
        # t=textrank(alldata["description"][0], withWeight=True)
        # print(type(plot))
        if type(plot) != str:
            plot = str(plot)
        plot = sub(r"[^\u4e00-\uf9a5]", " ", plot)
        keys = [str(i) for i in extract_tags(plot, withWeight=True)]
        # return new keywords as list, ignoring scores
        return keys

    def bag_words(x):
        # print(x)
        return(' '.join(x['plotwords']))

    alldata = pd.read_excel(
        './graduate_project/data/original_data/csv/hualien.xlsx')
    # Apply function to generate keywords
    alldata['plotwords'] = alldata['description'].apply(get_keywords)
    alldata['plotwords'] = alldata['plotwords'].fillna('')
    df_keys = pd.DataFrame()
    df_keys['title'] = alldata['spot']
    df_keys['keywords'] = alldata.apply(bag_words, axis=1)
    df_keys['keywords'] = df_keys['keywords'].fillna('')
    df_keys.to_csv(
        r'./graduate_project/data/original_data/csv/recommend_hualien.csv')


def model():
    try:
        df_keys = pd.read_csv(
            './graduate_project/data/original_data/csv/recommend_hualien.csv')
    except FileNotFoundError:
        addkeywords()
        pass
    df_keys = pd.read_csv(
        './graduate_project/data/original_data/csv/recommend_hualien.csv')
    # 將文件中的詞語轉換為詞頻矩陣
    cv = CountVectorizer()
    # 統計每個單字出現的次數
    df_keys['keywords'] = df_keys['keywords'].fillna('')
    cv_mx = cv.fit_transform(df_keys['keywords'])
    # create cosine similarity matrix
    cosine_sim = cosine_similarity(cv_mx, cv_mx)
    indices = pd.Series(df_keys.index, index=df_keys['title'])
    return cosine_sim, indices, df_keys


def recommend(title, n=10):
    cosine_sim, indices, df_keys = model()
    # retrieve matching movie title index
    if title not in indices.index:
        print("not in database.")
        return
    else:
        idx = indices[title]

    # cosine similarity scores of movies in descending order
    scores = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

    # top n most similar movies indexes
    # use 1:n because 0 is the same movie entered
    n = min(len(df_keys), n+1)
    top_n_idx = list(scores.iloc[1:n].index)
    return df_keys['title'].iloc[top_n_idx]


if __name__ == '__main__':
    for i in recommend('東大門夜市', n=5):
        print(i)
