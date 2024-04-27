import streamlit as st


def recommend():

    st.title("추천 방법")

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

    st.image("https://image.ibb.co/jYWZp9/wr.png", width=400)

    st.markdown(
        """
v : the number of votes for the movie

m : the minimum votes required to be listed in the chart

R : the average rating of the movie

C : the mean vote across the whole report
    """
    )
