import streamlit as st
from utils import *
import pandas_datareader as web
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np


selection_map = {
    'Bitcoin':(BitCoinModel, 'BTC-USD'),
    'Dogecoin':(DogeModel,'DOGE-USD'),
    'ETH':(ETHModel,'ETH-USD')
}

st.title("Simple stock prediction")
st.markdown("Here we are using LSTM neral network in order to predict the pice of cryptocurrency. Currently, we have a demostration of 3 types of cryptocurrency: Bitcoin, Doge and ETH.")

with st.form("my_form"):
    name_cryptocurrency = choice = st.selectbox(

    'Select the items you want?',

    ('Bitcoin', 'Dogecoin', 'ETH'))
    no_days = st.number_input('How many days do you want us to predict?', step=1)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        selected_model, code_cryptocurrency = selection_map[name_cryptocurrency]

        #get the data online
        end = dt.datetime.now()
        start = end - dt.timedelta(days = 31)
        df = web.DataReader(code_cryptocurrency, 'yahoo', start, end)
        df = df[['Open', 'High', 'Low', 'Close',	'Volume']]

        #run model
        prediction = selected_model.predict(df, no_days)
        prediction.insert(0, 'date', [(dt.datetime.now()+dt.timedelta(days = x)).strftime('%d/%m/%Y') for x in range(1,no_days+1)])

        st.markdown(f"The table below is the prediction of {name_cryptocurrency} in the next {no_days} days.")
        st.write(prediction)
        
        fig, ax = plt.subplots()
        plt.figure(figsize = ( 15 , 5 ))
        plt.title('Open price prediction')
        ax.plot(range(30),df['Open'].values[-30:], color='blue', label='Truth value')
        ax.plot(range(29, 30+ no_days), np.concatenate([df['Open'].values[-1:], prediction['Open'].values]), color='red', label='Predicted value')
        ax.legend(['Truth value (in the past)', f'Predicted value (in the next {no_days} days)'])
        plt.xlabel('Date',)
        plt.ylabel('Open Price ($)')

        st.pyplot(fig)

       