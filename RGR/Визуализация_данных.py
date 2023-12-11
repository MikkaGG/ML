import streamlit as st
from PIL import Image
import seaborn as sns
import pandas as pd
import plotly.express as px

st.subheader('Визуализации признаков данных')
X_data = pd.read_csv("./data/weather_data1.csv")

char_select = st.selectbox(
    label='Выберите тип визуализации данных',
    options=['Диаграмма рассеяния', 'Линейная диаграмма', 'Гистограмма', 'Коробочный график']
)

numeric_columns = list(X_data.columns)

if char_select == 'Диаграмма рассеяния':
    x_value = st.selectbox('Ось x: ', options=numeric_columns)
    y_value = st.selectbox('Ось y: ', options=numeric_columns)
    title = 'Диаграмма рассеяния признака ' + x_value + ' и признака ' + y_value
    plot1 = px.scatter(X_data, x=x_value, y=y_value, title=title)
    st.plotly_chart(plot1)
if char_select == 'Линейная диаграмма':
    x_value = st.selectbox('Ось x: ', options=numeric_columns)
    y_value = st.selectbox('Ось y: ', options=numeric_columns)
    title = 'Линейная диаграмма зависимости признака ' + x_value + ' и признака ' + y_value
    plot2 = px.line(X_data, x=x_value, y=y_value, title=title)
    st.plotly_chart(plot2)
if char_select == 'Гистограмма':
    x_value = st.selectbox('Выберите признак: ', options=numeric_columns)
    title = 'Гистограмма признака ' + x_value
    plot3 = px.line(X_data, x=x_value, title=title)
    st.plotly_chart(plot3)
if char_select == 'Коробочный график':
    x_value = st.selectbox('Ось x: ', options=numeric_columns)
    y_value = st.selectbox('Ось y: ', options=numeric_columns)
    title = 'Коробочный график зависимости признака ' + x_value + ' и признака ' + y_value
    plot4 = px.box(X_data, x=x_value, y=y_value, )
    st.plotly_chart(plot4)