import streamlit as st
import pickle
from tmdbv3api import Movie, TMDb
import os

movie = Movie()
tmdb = TMDb()
# tmdb.api_key = "dd3249e857ef4f461dd593504b320009"
tmdb.language = "ko-KR"


# 영화의 제목을 입력받으면 코사인 유사도를 통해 가장 유사도가 높은 상위 10개의 영화 목록 반환
def get_recommendations(title):
    # 영화 제목을 통해서 전체 데이터 기준 그 영화의 index 값을 얻기
    idx = movies[movies["title"] == title].index[0]

    # 코사인 유사도와 매트릭스(cosine_sim)에서 idx에 해당하는 데이터를 (idx, 유사도)) 형태로 얻는다
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 코사인 유사도 기준으로 내림차순 정렬
    # key 를 통하여 정렬할 기준을 정할 수 있다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 자기 자신을 제외한 10개의 추천 영화를 슬라이싱
    sim_scores = sim_scores[1:11]

    # 추천 영화 목록 10개의 인덱스 정보 추출
    movie_indicies = [i[0] for i in sim_scores]

    # 인덱스 정보를 통해 영화 제목 추출
    movie_images = []
    movie_titles = []
    for i in movie_indicies:
        movie_id = movies["id"].iloc[i]
        details = movie.details(movie_id)

        image_path = details["poster_path"]
        if image_path:
            image_path = "https://image.tmdb.org/t/p/w500" + image_path
        else:
            image_path = "media/no_image.jpg"

        movie_images.append(image_path)
        movie_titles.append(details["title"])
    return movie_images, movie_titles


movies = pickle.load(open("movie_recommend/movies.pickle", "rb"))
cosine_sim = pickle.load(open("movie_recommend/cosine_sim2.pickle", "rb"))


st.set_page_config(
    page_title="movie recommendation",
    page_icon="🎬",
    # layout="wide",
)

# tmdb.api_key = "dd3249e857ef4f461dd593504b320009"
with st.sidebar:
    tmdbkey = None
    tmdbkey = st.text_input("Write Your TMDB API key: ", type="password")
    os.environ["TMDB_API_KEY"] = tmdbkey
    # tmdb.api_key = tmdbkey

st.markdown(
    """
        # Movie Recommend
    """
)

if tmdbkey:
    # tmdb.api_key = tmdbkey
    movie_list = movies["title"].values
    title = st.selectbox("좋아하는 영화를 선택해주세요", movie_list)
    if st.button("Recommend"):
        with st.spinner("잠시만 기다려주세요"):
            images, titles = get_recommendations(title)

            idx = 0
            for i in range(0, 2):
                cols = st.columns(5)
                for col in cols:
                    col.image(images[idx])
                    col.write(titles[idx])
                    idx += 1
else:
    st.markdown(
        """
                Welcome to Movie Recommend, choose or write movie name.
                
                Start by writing TNDB API key in the sidebar.
                """
    )
