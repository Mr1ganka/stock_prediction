import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pandas_datareader.data as data
#import pandas_datareader.data as web
from datetime import datetime, date, timedelta  
import os.path
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler

import streamlit as st

st.title('Stock Trend Predictor')

def get_historical(quote):  #function to create ticker.csv
        end = datetime.now()
        start = datetime(end.year-9,end.month,end.day)
        data = yf.download(quote, start=start, end=end)
        df = pd.DataFrame(data=data)
        df.to_csv(''+quote+'.csv')
        if(df.empty):
            ts = TimeSeries(key='N6A6QT6IBFJOPJ70',output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol='NSE:'+quote, outputsize='full')
            #Format df
            #Last 2 yrs rows => 502, in ascending order => ::-1
            data=data.head(503).iloc[::-1]
            data=data.reset_index()
            #Keep Required cols only
            df=pd.DataFrame()
            #df['Date']=data['date']
            df['Open']=data['1. open']
            df['High']=data['2. high']
            df['Low']=data['3. low']
            df['Close']=data['4. close']
            #df['Adj Close']=data['5. adjusted close']
            df['Volume']=data['6. volume']
            df.drop(['Adj Close'])
            df.to_csv(''+quote+'.csv',index=False)
            #df.drop(['Date','Adj Close'], axis =1)
        return (end.date(),start.date())

ticker_symbol=st.text_input("Enter Stock Ticker", "AAPL")
end,start=get_historical(ticker_symbol)

#Data Description
st.subheader('Data from '+str(start)+" to "+str(end))
df = pd.read_csv(ticker_symbol+'.csv')
st.write(df.describe())

#Plotting the Chart for Closing Vs Time
st.subheader("Closing vs Time chart")
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

#Calculating Moving Average of 100 and 200 days
ma100=df.Close.rolling(100).mean() #created a moving average of ma100
#ma100
ma200=df.Close.rolling(200).mean() #created a moving average of ma200
#ma200

st.subheader("Rolling Average of 100 and 200 days visualised")
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')
st.pyplot(fig)

end_limit=len(df)
limit=int(end_limit*0.7)
data_training = pd.DataFrame(df['Close'][0:limit])
data_testing =  pd.DataFrame(df['Close'][limit:end_limit])

#LSTM model uses Data Range between 0 and 1 SCALING THE DATA
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
#created an Object to normalize data between the two

data_testing_array=scaler.fit_transform(data_testing)
data_training_array=scaler.fit_transform(data_training)
#data_training_array.shape
#transforming the columns to feature range

#separating the data for x_train and Y_train
x_train=[]
y_train=[]

for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
    
x_train,y_train = np.array(x_train), np.array(y_train)

import os.path

if(os.path.exists('keras_model.h5')):
    model=load_model('keras_model.h5')
else:
     ####Creating the Model###########
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True,
                input_shape =(x_train.shape[1],1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=1))
    #model.summary()
    #########   model creation over #####################
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train,y_train, epochs=50)
    model.save('keras_model.h5') #saving the model to avoid re-compiling everytime

past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing, ignore_index=True)
input_data =scaler.fit_transform(final_df)

x_test=[] 
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
#the scale factor for the 
#print(scaler.scale_)
scale=scaler.scale_[0]
scale_factor=1/scale
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

#plotting thryy
st.subheader("Original vs Predicted")
fig=plt.figure(figsize=(12,6))
plt.plot(y_predicted,'r', label='Predicted Price')
plt.plot(y_test,'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
#plt.show()
st.pyplot(fig)

################## Testing out the Model ######################
st.subheader("Testing the model")

#setting end to three days back
today = date.today()
day_before_yesterday = date.today() - timedelta(days=3)
#end

#end = datetime.now()
#getting data worth of last 2 years
start = datetime(end.year-2,end.month,end.day)
data = yf.download(ticker_symbol, start=start, end=day_before_yesterday)
st.subheader("Gathering data for testing")
st.dataframe(data[-60:])
#getting the data and filtering it upon just the Closing price of the day
data=data.filter(['Close'])

#filtering the data based upon just the last 60 days
few=60
last_few_days = data[-few:].values
#last_60_days

st.subheader("Gathered data for the last "+str(few)+ "days till: "+str(day_before_yesterday))
#st.dataframe(last_few_days)
#scaling the last 60days value
last_few_days_scaled = scaler.transform(last_few_days)
X_test=[]
X_test.append(last_few_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
#print(last_60_days_scaled)

#Get the predicted scaled price
pred_price = model.predict(X_test)#getting the original price
pred_price = scaler.inverse_transform(pred_price)
st.subheader("The model has predicted the price for yesterday as" )
st.code(pred_price[0][0])


#setting end to two days back
today = date.today()
day_before_yesterday = date.today() - timedelta(days=2)
#end
#Yesterdays actual price
yesterday = date.today() - timedelta(days=1)
st.subheader("The actual price for yesterday was" )
data = yf.download(ticker_symbol, start=day_before_yesterday, end=yesterday)

#data=data.filter(['Close'])
st.code(data['Close'][0])