import streamlit as st
from tmdbv3api import Movie, TMDb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import linear_kernel

movie_code = Movie()
tmdb = TMDb()
tmdb.language = "ko-KR"

tmdb_credits = pd.read_csv("movie_recommend/tmdb_5000_credits.csv")
tmdb_movies = pd.read_csv("movie_recommend/tmdb_5000_movies.csv")

tmdb_credits.columns = ["id", "title", "cast", "crew"]
tmdb_movies = tmdb_movies.merge(tmdb_credits[["id", "cast", "crew"]], on="id")

# 코드: 인구 통계학적 필터링
m = tmdb_movies["vote_count"].quantile(0.9)
C = tmdb_movies["vote_average"].mean()
q_movies = tmdb_movies.copy().loc[tmdb_movies["vote_count"] >= m]


def weighted_rating(x, m=m, C=C):
    v = x["vote_count"]
    R = x["vote_average"]
    return (v / (v + m) * R) + (m / (m + v) * C)


q_movies["score"] = q_movies.apply(weighted_rating, axis=1)
q_movies = q_movies.sort_values("score", ascending=False)
top_movie = q_movies[["id", "title", "vote_count", "vote_average", "score"]].head(10)

# 코드: 컨텐츠 기반 필터링 - 줄거리


def recommend_1():

    st.title("추천 방법")
    st.text(
        "source: https://www.kaggle.com/code/ibtesama/getting-started-with-a-movie-recommendation-system"
    )

    with st.container():
        col1, col2 = st.columns(2)

        col1.download_button(
            label="Download tmdb_5000_credits.csv",
            data="movie_recommend/tmdb_5000_credits.csv",
            file_name="tmdb_5000_credits.csv",
        )
        col2.download_button(
            label="Download tmdb_5000_movies.csv",
            data="movie_recommend/tmdb_5000_movies.csv",
            file_name="tmdb_5000_movies.csv",
        )

    # 1. 인구 통계확적 필터링
    st.subheader("인구 통계학적 필터링(Demographic Filtering)")
    st.text(":많은 사람들이 일반적으로 좋아하는 아이템 추천")

    st.image("https://image.ibb.co/jYWZp9/wr.png", width=400)

    st.markdown(
        """
v : the number of votes for the movie

m : the minimum votes required to be listed in the chart

R : the average rating of the movie

C : the mean vote across the whole report
    """
    )

    code_demographic = """
        import pandas as pd
        import numpy as np
        
        tmdb_credits = pd.read_csv('tmdb_5000_credits.csv)
        tmdb_movies = pd.read_csv('tmdb_5000_movies.csv)
        
        tmdb_movies = tmdb_movies.merge(tmdb_credits [['id', 'cast', 'crew']], on= 'id')
        
        
        m = tmdb_movies['vote_count'].quantile(0.9)
        # the minimum votes required to be listed in the chart
        # 상위 10%에 대한 데이터를 뽑는다. 
        
        C = tmdb_movies['vote_average'].mean()
        # the mean vote across the whole report
        
        q_movies = tmdb_movies.copy().loc[tmdb_movies['vote_count'] >= m]
        
        
        def weighted_rating(x, m=m, C=C):
            v = x['vote_count'] # the number of votes for the movie
            R = x['vote_average'] # the average rating of the movie
            return (v / (v+m) * R) + (m / (m+v) * C)
            
        q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
        
        q_movies = q_movies.sort_values('score', ascending=False)
        """
    st.code(code_demographic, language="python")

    if st.button("Recommend"):
        with st.spinner("잠시만 기다려주세요"):
            images, titles = get_recommendations_ten(top_movie)

            idx = 0
            for i in range(0, 2):
                cols = st.columns(5)
                for col in cols:
                    col.image(images[idx])
                    col.write(titles[idx])
                    idx += 1

    # 컨텐츠 기반 필터링: 줄거리 기반
    st.subheader("컨텐츠 기반 필터링(Content Based Filtering)")
    st.text(":특정 아이템과 유사한 아이템 추천")
    st.subheader("줄거리 기반(overview)")

    code_content_overview = """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        from sklearn.metrics.pairwise import linear_kernel
        
        #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
        tfidf = TfidfVectorizer(stop_words='english')

        #NaN 값이 있는지 있는지 확인 -> True면 NaN 값이 있다.
        tmdb_movies['overview'].isnull().values.any()

        #Replace NaN with an empty string
        tmdb_movies['overview'] = tmdb_movies['overview'].fillna('')
        tmdb_movies['overview'].isnull().values.any()
        
        #Construct the required TF-IDF matrix by fitting and transforming the data
        tfidf_matrix = tfidf.fit_transform(tmdb_movies['overview'])
        tfidf_matrix.shape
        
        # linear_kernel을 통한 유사도 측정
        # linear_kernel : 코사인 유사도를 구하는 방법
        # 코사인 유사도: 두 벡터 간의 코사인 각도를 이용하여 구할 수 있는 두 벡터의 유사도, -1 이상 1 이하의 값을 가지며 값이 1에 가까울수록 유사도가 높다.
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) # 코사인 유사도
        
        
        # 영화의 제목을 입력받으면 코사인 유사도를 통해 가장 유사도가 높은 상위 10개의 영화 목록 반환
        def get_recommendations(title, cosine_sim=cosine_sim):
            # 영화 제목을 통해서 전체 데이터 기준 그 영화의 index 값을 얻기
            idx = indices[title]
            
            # 코사인 유사도와 매트릭스(cosine_sim)에서 idx에 해당하는 데이터를 (idx, 유사도)) 형태로 얻는다
            sim_scores = list(enumerate(cosine_sim[idx]))
            
            # 코사인 유사도 기준으로 내림차순 정렬
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # 자기 자신을 제외한 10개의 추천 영화를 슬라이싱
            sim_scores = sim_scores[1:11]
            
            # 추천 영화 목록 10개의 인덱스 정보 추출
            movie_indicies = [i[0] for i in sim_scores]
            
            # 인덱스 정보를 통해 영화 제목 추출
            return df2['title'].iloc[movie_indicies]
            
        get_recommendations('The Avengers') # 영화 추천
        """
    st.code(code_content_overview, language="python")


# 보여주기
# 인구 통계학적 필터링
def get_recommendations_ten(top_movie):
    # 인덱스 정보를 통해 영화 제목 추출
    movie_images = []
    movie_titles = []
    for i in range(10):
        movie_id = top_movie["id"].iloc[i]
        details = movie_code.details(movie_id)

        image_path = details["poster_path"]
        if image_path:
            image_path = "https://image.tmdb.org/t/p/w500" + image_path
        else:
            image_path = "media/no_image.jpg"

        movie_images.append(image_path)
        movie_titles.append(details["title"])
    return movie_images, movie_titles


# 컨텐츠 기반 필터링 - 줄거리 기반
