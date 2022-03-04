import csv
import streamlit as st
from keras.models import load_model
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from sklearn.metrics import mean_squared_error


def app():   
    
    st.title("Forecast Measles Cases")
    weektoforecast = st.sidebar.number_input('Enter Number of Weeks to forecast',min_value=4)
    cases=st.sidebar.number_input('Enter Number of Cases For The Current Day for Measles',value=0)


    
    #This Section is for prediction on test data

    n_features=1

    length=50

    df=pd.read_csv("Measles.csv")

    df['Start Date'] = pd.to_datetime(df['Start Date'], format='%d/%m/%Y')

    df.set_index('Start Date',inplace=True)

    test_weeks=50
    test_index=len(df)-test_weeks
    train=df.iloc[:test_index]
    test=df.iloc[test_index:]


    scaler_test=MinMaxScaler()
    loaded_model=load_model('measles_finalmodel.h5')
    test_scaled=scaler_test.fit_transform(test)

    test_predictions = []
    first_eval_batch = test_scaled[-length:]
    current_batch = first_eval_batch.reshape((1, length, n_features))

    for i in range(len(test)):
        current_pred = loaded_model.predict(current_batch)[0]
        test_predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    true_predictions = scaler_test.inverse_transform(test_predictions)
    test['Predictions']=true_predictions


    

    



    #This Section is For Forecast
    input_dataframe_for_forecast=pd.DataFrame(data={'Case':[cases]})

    forecast_scaler=MinMaxScaler()

    forecast_scaled=forecast_scaler.fit_transform(input_dataframe_for_forecast)

    forecast = []
    howmanyweeks = weektoforecast
    length_forecast=len(input_dataframe_for_forecast)
    n_features=1

    first_eval_batch = forecast_scaled[-length_forecast:]
    current_batch = first_eval_batch.reshape((1, length_forecast, n_features))

    for i in range(howmanyweeks):
        current_pred = loaded_model.predict(current_batch)[0]
        forecast.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    forecast=forecast_scaler.inverse_transform(forecast)

    now=datetime.now()
    current_time = now.strftime("%d/%m/%Y")


    forecast_index=pd.date_range(current_time,periods=howmanyweeks,freq='W')
    forecast_df=pd.DataFrame(data=forecast,index=forecast_index,columns=['Forecast'])
    print(forecast_df)

    st.header("Forecast Data")
    st.dataframe(forecast_df)


    st.header("Forecasted Data Graph")
    st.line_chart(forecast_df)

 





