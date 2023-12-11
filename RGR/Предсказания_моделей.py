import streamlit as st
import pickle
import numpy as np
import sklearn
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json

st.subheader('Здесь можно получить предсказание моделей')

ModelKnn = pickle.load(open('./models/ModelKnn.sav', 'rb'))
ModelStacking = pickle.load(open('./models/ModelStacking.sav', 'rb'))
json_file = open('./models/model_classification.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
ModelNetwork = model_from_json(loaded_model_json)
ModelNetwork.load_weights("./models/model_weights.h5")

def prediction(input_data, nameModel):
    numpy_data= np.asarray(input_data, dtype = float)
    input_reshaped = numpy_data.reshape(1, -1)
    if nameModel == 'K Neighbors Classifier':
        prediction = ModelKnn.predict(input_reshaped)
    if nameModel == 'Stacking Classifier':
        prediction = ModelStacking.predict(input_reshaped)
    if nameModel == 'ModelNetwork':
        prediction = ModelNetwork.predict(input_reshaped)
    if (prediction == 0):
        st.success('Дождя не будет')
    else:
        st.warning('Дождь будет')

st.markdown('Перед заполенением необходимо ознакомится с признаковым пространством на странице *Информация о данных*.')

MinTemp = st.text_input('MinTemp (в градусах Цельсия)')
MaxTemp = st.text_input('MaxTemp (в градусах Цельсия)')
Rainfall = st.text_input('Rainfall (в мм)')
WindGustSpeed = st.text_input('WindGustSpeed (в км/ч)')
WindSpeed9am = st.text_input('WindSpeed9am (в км/ч)')
WindSpeed3pm = st.text_input('WindSpeed3pm (в км/ч)')
Humidity9am = st.text_input('Humidity9am (в процентах)')
Humidity3pm = st.text_input('Humidity3pm (в процентах)')
Pressure9am = st.text_input('Pressure9am (в гектопаскалях)')
Pressure3pm = st.text_input('Pressure3pm (в гектопаскалях)')
Cloud3pm = st.text_input('Cloud3pm (в октах)')
Temp9am = st.text_input('Temp9am (в градусах Цельсия)')
Temp3pm = st.text_input('Temp3pm (в градусах Цельсия)')
RainTomorrow = st.text_input('RainTomorrow (в мм)')
Day = st.text_input('Day')

select_event1 = st.selectbox('Выберите модель для предсказания', ('K Neighbors Classifier', 'Stacking Classifier', 'ModelNetwork'))
result = ''
if st.button('Результат'):
    result = prediction([MinTemp, MaxTemp, Rainfall, WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am,
    Pressure3pm, Cloud3pm, Temp9am, Temp3pm, RainTomorrow, Day], select_event1)
