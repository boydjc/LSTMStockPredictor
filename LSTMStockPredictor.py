import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, LSTM
from keras.utils import to_categorical
import os
from datetime import datetime, timedelta
import bs4
import requests
import json

def getYahooData(ticker, startDate, endDate, interval):
    
    #get todays date
    
    today = datetime.now().date()
    
    tomorrow = today + timedelta(days=1)
    
    today = today.strftime('%d-%m-%Y')
    tomorrow = tomorrow.strftime('%d-%m-%Y')

    
    # have to convert each date to unix for the yahoo url
    # format for date will be 'DD-MM-YYYY'
    
    if startDate == endDate:
        startDateUnix = int(datetime.strptime(today, '%d-%m-%Y').timestamp())
        endDateUnix = int(datetime.strptime(tomorrow, '%d-%m-%Y').timestamp())
    else:
        startDateUnix = int(datetime.strptime(startDate, '%d-%m-%Y').timestamp())
        endDateUnix = int(datetime.strptime(endDate, '%d-%m-%Y').timestamp())
    
    url = 'https://query1.finance.yahoo.com/v8/finance/chart/' + ticker + '?symbol=' + ticker + '&period1=' + str(startDateUnix) + '&period2=' + str(endDateUnix) + '&interval=' + str(interval)

    res = requests.get(url, headers={"User-Agent":"Mozilla/5.0"})

    soup = bs4.BeautifulSoup(res.text, 'html.parser')
    
    # sort through the nested dictionary mess and put each value in its own list
    
    newDictionary=json.loads(str(soup))
    newDictionary2 = newDictionary['chart']['result'][0]
    
    unixDates = newDictionary2['timestamp']
    
    dates = []
    
    # turn the unix timestamps back into dates
    for item in unixDates:
        dateFormat = datetime.fromtimestamp(item).date().strftime('%Y-%m-%d')
        dates.append(dateFormat)
    
    newDictionary3 = newDictionary2['indicators']['quote'][0]
    
    # convert each value into float so that we can only keep two places after the decimal
    
    strOpens = newDictionary3['open']
    
    opens = []
    
    for item in strOpens:
        if item == None:
            number = 0.00
            opens.append(number)
        else:
            number = round(float(item), 2)
            opens.append(number)
        
        
    strHighs = newDictionary3['high']
    
    highs = []
    
    for item in strHighs:
        if item == None:
            number = 0
            highs.append(number)
        else:
            number = round(float(item), 2)
            highs.append(number)
        
    strLows = newDictionary3['low']
    
    lows = []
    
    for item in strLows:
        if item == None:
            number = 0
            lows.append(number)
        else:
            number = round(float(item), 2)
            lows.append(number)
        
    strCloses = newDictionary3['close']
    
    closes = []
    
    for item in strCloses:
        if item == None:
            number = 0
            closes.append(number)
        else:
            number = round(float(item), 2)
            closes.append(number)
        
    strVolumes = newDictionary3['volume']

    volumes = []
    
    for item in strVolumes:
        if item == None:
            number = 0
            volumes.append(number)
        else:
            number = int(item)
            volumes.append(number)
            
    # only worry about adj. close if there is an interval 1d or more
    # have to make a new dictionary just for the adjClose prices

    if interval != '1m' and interval != '2m' and interval != '5m':
    
        newDictionary4 = newDictionary2['indicators']['adjclose'][0]
    
        strAdjCloses = newDictionary4['adjclose']
    
        adjCloses = []
    
        for item in strAdjCloses:
            if item == None:
                number = 0
            else:
                number = round(float(item), 2)
                adjCloses.append(number)
        
    # append them all to their own list so that each entry in the list is a full OHLC entry for the day
    
    newList = []
    finalList = []
    
    for i in range(0, len(dates)):
        newList.append(dates[i])
        newList.append(opens[i])
        newList.append(highs[i])
        newList.append(lows[i])
        newList.append(closes[i])
        if interval != '1m' and interval != '2m' and interval != '5m':
            newList.append(adjCloses[i])
        newList.append(volumes[i])
        finalList.append(newList)
        newList = []
     
    return finalList

def grabData():
    stopData = False
    stockData = []

    ticker = input("ticker:")
    tInterval = input("Time Interval(1m, 2m, 5m, 1d, 1w, 1m):")

    while(not stopData):

        startDate = input("Start Date:")
        endDate = input("End Date:")
        
        # get stock data
        newData = getYahooData(ticker, startDate, endDate, tInterval)

        for count in range(0, len(newData)-1):
            stockData.append(newData[count])

        print('Data Retrieved.')
        print("Continue Grabbing Data? (y/n)")
        getMoreData = input(":")

        if getMoreData == 'n':
            break;

        os.system('cls')

    return[stockData, tInterval]


def makePrediction():

    data = grabData()

    # data[0] = stock data, data[1] = time interval
    if data[1] != '1m' and data[1] != '2m' and data[1] != '5m':
        stockData = pd.DataFrame(data[0], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj. Close', 'Volume'])
    else:
        stockData = pd.DataFrame(data[0], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
    del(stockData['Date'])
    del(stockData['Open'])
    del(stockData['High'])
    del(stockData['Low'])
    del(stockData['Volume'])

    if data[1] != '1m' and data[1] != '2m' and data[1] != '5m':
        del(stockData['Adj. Close'])

    # 90% train, 10% test

    trainSize = int(len(stockData) * .9)

    trainStockData = stockData[:trainSize]
    testStockData = stockData[trainSize:]

    # UNCOMMENT FOR SMA PREDICTION
    #trainStockData['SMA'] = trainStockData['Close'].rolling(window=5).mean()
    #trainStockData = trainStockData.fillna(0)
    #del(trainStockData['Close'])
    trainStockData = np.array(trainStockData)

    # UNCOMMENT FOR SMA PREDICTION
    #testStockData['SMA'] = testStockData['Close'].rolling(window=5).mean()
    #testStockData = testStockData.fillna(0)
    #del(testStockData['Close'])
    testStockData = np.array(testStockData)

    scaler = MinMaxScaler(feature_range=(-1, 1))

    # train data

    x_train = []
    y_train = []

    for count in range(0, len(trainStockData)):
        if count % 2 == 0:
            x_train.append(trainStockData[count])
        else:
            y_train.append(trainStockData[count])

    x_train = np.array(x_train)
    x_train = scaler.fit_transform(x_train)
    x_train = x_train.reshape(-1, 1, 1)

    y_train = np.array(y_train)
    y_train = scaler.fit_transform(y_train)
    y_train = y_train.reshape(-1, 1, 1)

    # make sure x_train and y_train are the same

    if(x_train.shape[0] != y_train.shape[0]):
        x_train = np.delete(x_train, len(x_train)-1, 0)

    # test data

    x_test = []
    y_test = []

    for count in range(0, len(testStockData)):
        if count % 2 == 0:
            x_test.append(testStockData[count])
        else:
            y_test.append(testStockData[count])

    x_test = np.array(x_test)
    x_test = scaler.fit_transform(x_test)
    x_test = x_test.reshape(-1, 1, 1)

    y_test = np.array(y_test)
    y_test = scaler.fit_transform(y_test)
    y_test = y_test.reshape(-1, 1, 1)


    # make sure x_train and y_train are the same

    if(x_test.shape[0] != y_test.shape[0]):
        x_test = np.delete(x_test, len(x_test)-1, 0)

    # build network

    network = Sequential()
    network.add(Dense(10000, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
    network.add(LSTM(1000, activation='relu', return_sequences=True))
    network.add(Dropout(.5))
    network.add(Dense(1))

    network.compile(optimizer='adam', loss='mse', metrics=['acc'])

    network.summary()
    history = network.fit(x_train, y_train, epochs=10, verbose=2)

    evaluation = network.evaluate(x_test, y_test)

    evalAcc = evaluation[1]

    # make a prediction on the last day for y_test

    actualData = y_test

    actualData = actualData.reshape(-1, 1)
    actualData = scaler.inverse_transform(actualData)

    newData = np.array(actualData[len(actualData)-1]) # last value in y_test which is now actualData
    previousValue = newData # store this value for later
    newData = newData.reshape(-1, 1)
    newData = scaler.fit_transform(newData)
    newData = newData.reshape(-1, 1, 1)

    prediction = network.predict(newData)
    prediction = prediction.reshape(-1, 1)
    prediction = scaler.inverse_transform(prediction)
    print('Previous Close: ', previousValue)
    print('Next Predicted Close: ', prediction)

    # add or subtract the accuracy to try and get a better predition 
    if prediction > previousValue:
        print('Adjusted Prediction: ', prediction + round(evalAcc, 2))
    else:
        print('Adjusted Prediction: ', prediction - round(evalAcc, 2))



if __name__ == '__main__':

    makePrediction();











