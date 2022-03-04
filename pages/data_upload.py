import collections
from numpy.core.defchararray import lower
import streamlit as st
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten



def app():
    st.title("Forecast Contagious Disease Dataset")
    st.header('Creating the model')
    st.caption('A model represents what was learned by a machine learning algorithm. LSTM is the algorithm that will be used in the study to predict the cases of disease.')
    uploaded_file = st.file_uploader("Choose a file")
    if st.button('Forecast'):
        st.header('Model Result')
        if uploaded_file is not None:
            dataframe= pd.read_csv(uploaded_file, usecols=[1], engine='python')
            dataframe = np.array(dataframe)
            
            df = pd.read_csv(uploaded_file.name)

            # preparing datastet
            def prepare_data(timeseries_data, n_features):
                X, y =[],[]
                for i in range(len(timeseries_data)):
                    # find the end of this pattern
                    end_ix = i + n_features
                    # check if we are beyond the sequence
                    if end_ix > len(timeseries_data)-1:
                        break
                        # gather input and output parts of the pattern
                    seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
                    X.append(seq_x)
                    y.append(seq_y)
                return np.array(X), np.array(y)

            #spliting the dataframe into 80% train and 20% test
            df_train = df.iloc[:int(len(df)*0.8),1]
            df_test = df.iloc[int(len(df)*0.8):,1]
            
            
            st.write("Train Set: "+str(len(df_train)))
            st.write("Test Set: "+str(len(df_test)))
            future_period = len(df_test)

            #preparing train and test datasets
            train = np.array(df_train)
            test = np.array(df_test)

            # choose a number of time steps
            n_steps = 10
            # split into samples
            X, y = prepare_data(train, n_steps)
            X_test, y_test = prepare_data(test, n_steps)

            # reshape from [samples, timesteps] into [samples, timesteps, features]
            n_features = 1
            X = X.reshape((X.shape[0], X.shape[1], n_features))

            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

            # define model
            model = Sequential()
            model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
            model.add(LSTM(30, activation='relu', return_sequences=True))
            model.add(LSTM(10, activation='relu', return_sequences=True))
            model.add(LSTM(5, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')

            #training the model on train data
            model.fit(X, y, epochs=30, verbose=1)

            #testing the model on the train data
            yp = model.predict(X)

            #testing the model on test data
            y_p = model.predict(X_test)

            # Forecasting For the next n days
            x_input = np.array(X[len(X)-1])
            x_input = x_input.reshape((1, n_steps, n_features))
            # y = model.predict(x_input)
            ypn=[]

            for i in range(future_period):
                k = model.predict(x_input)
                ypn.append(k[0,0])
                x_input = np.delete(x_input, (0,0,0))
                x_input = np.append(x_input, k)
                x_input = x_input.reshape((1, n_steps, n_features))

            # Visualization
            st.subheader('Visualization')
            day_new=np.arange(len(df_train) - n_steps)
            day_pred = np.arange(len(df_train) - n_steps,len(df_train) +len(df_test) - 2*n_steps )
            day_forecast=np.arange(len(df_train) - n_steps,len(df_train) +future_period - n_steps )
            fig = plt.figure(figsize=(12, 6))
            plt.plot(day_new,y, '-g',label= "Actual")
            plt.plot(day_pred, y_p, 'bo', label= 'test')
            plt.plot(day_forecast,ypn, '-r', label= "Forecasted")
            plt.legend(loc="upper left")
            plt.xlabel("Days")
            plt.ylabel("Cases")
            st.pyplot(fig)

            # RMSE
            from sklearn.metrics import mean_squared_error
            rms = mean_squared_error(y, yp, squared=False)
            print(f"RMSE: {rms}")
            st.write("RMSE: "+str(rms))

            model.save("timeseries.h5")

            
            # Table
            st.subheader('Table')
            df_f = pd.DataFrame({'Date': df.iloc[10:len(df_train),0] , 'actual':y ,'predicted': yp.reshape(-1)})
            date = []
            for i in range(len(df_test)-n_steps):
                d = pd.to_datetime(pd.to_datetime(df_f.iloc[len(df_f) -1,0])) + pd.to_timedelta((i+1)*7, unit='D')
                date.append(d)
            df_ff = pd.DataFrame({'Date': date , 'actual':y_test ,'predicted': y_p.reshape(-1)})
            disease_data = pd.concat([df_f, df_ff], ignore_index=True)

            st.subheader('Train Set Forecasted Result')   
            st.dataframe( df_f)

            st.subheader('Test Set Forecasted Result')   
            st.dataframe( df_ff)

            model.save("model.h5")
            
        else:
            st.write('UPLOAD CSV FILE')