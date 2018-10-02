from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os, math
import keras.backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('fivethirtyeight')
#print(style.available)

def loadData(link):
    data = pd.read_csv(link)
    datetimeColName = data.columns[0]
    data[datetimeColName] = pd.to_datetime(data[datetimeColName])
    data.set_index(datetimeColName, inplace=True)
    data.sort_index()
    return data

def scale_to_0and1(data):
    scaler = MinMaxScaler(feature_range=(0, 1))  # MinMax Scaler
    data = scaler.fit_transform(data)  # input: ndarray type data
    return(data, scaler)


# split into train and test sets
def splitData(data, trainPercent=0.7, split_by_time=False, split_date=None):
    if split_by_time is False:
        train_size = int(len(data) * trainPercent)
        train_data, test_data = data.iloc[0:train_size,], data.iloc[train_size:,]
        print("\n", "train length:", len(train_data),"\n", "test length:", len(test_data))
    elif split_by_time is True:
        # split_date = pd.Timestamp("01-01-2011")
        split_date = split_date
        train_data = data.ix[:split_date, :]
        train_data.drop(split_date, axis=0, inplace=True)
        test_data = data.ix[split_date:, :]
    return(train_data, test_data)



def reshapeForLSTM(data, time_steps=None):
    """
    :param data: intput data
    :param time_steps: time steps after
    :return: reshaped data for LSTM
    """
    """
    The LSTM network expects the input data (X) 
    to be provided with 
    a specific array structure in the form of: 
    [samples, time steps, features].
    """
    if time_steps is None:
        print("please denote 'time_steps'...!")
        return(None)
    else:
        data_reshaped = np.reshape(data, (data.shape[0], time_steps, 1))
    return(data_reshaped)




# --- create dataset with window size --- #
def sequentialize(scaled_inputData, inputData_index, window_size=None, to_ndarray=False):
    if window_size is None:
        print("\n", "please use 'window_size'...!")
        return(None)
    elif isinstance(window_size, int):
        # change type to use 'shift' of pd.DataFrame
        scaled_inputData = pd.DataFrame(scaled_inputData, columns=["value"], index=inputData_index)

        # dataframe which is shifted as many as window size
        for idx in range(1,window_size+1):
            scaled_inputData["before_{}".format(idx)] = scaled_inputData["value"].shift(idx)

        # drop na
        inputSequence = scaled_inputData.dropna().drop('value', axis=1)
        output = scaled_inputData.dropna()[['value']]
        
        if to_ndarray is False:
            return(inputSequence, output)
        else:
            inputSequence = inputSequence.values

            output = output.values
            return(inputSequence, output)



### training with different params sets (epochs + batch size). and save error scores.
### We can choose smallest error score params set to apply to model
def optimizing(X_train, y_train,X_test, y_test, epochs, batch_size):
    error_scores = pd.DataFrame(columns=['epochs','batch_size','RMSE'])
    for e in epochs:
        for b in batch_size:
            # --- train LSTM network --- #
            # todo: what "K.clear_session()" do...?
            K.clear_session()
            model = Sequential()
            model.add(LSTM(units=20, input_shape=(12, 1)))  # (timestep, feature)
            model.add(Dense(units=1))  # output = 1
            model.compile(loss="mean_squared_error", optimizer="adam")
            model.summary()
            # todo: inspect "EarlyStopping function()"
            early_stop = EarlyStopping(monitor="loss", patience=1, verbose=1)
            # todo: make loss list & accumulate loss value to loss list
            # todo: & plot lost list
            model.fit(X_train, y_train, epochs=e, batch_size=b, verbose=2, callbacks=[early_stop])
            # todo: check 'why full iteration is stopped'
            # --- predict LSTM network --- #
            # todo: fix codes below...!
            trainPred = model.predict(X_train)
            trainPred = scaler.inverse_transform(trainPred)
            trainY = scaler.inverse_transform(y_train)
            testPred = model.predict(X_test)
            testPred = scaler.inverse_transform(testPred)
            testY = scaler.inverse_transform(y_test)
            # --- MSE --- #
            trainScore = math.sqrt(mean_squared_error(trainY, trainPred))
            testScore = math.sqrt(mean_squared_error(testY, testPred))
            print("\n", "Train Score:   %.1f RMSE" % (trainScore), "\n", " Test Score:   %.1f RMSE" % (testScore))
            error_scores = error_scores.append([{'epochs':e,'batch_size':b,'RMSE':testScore}], ignore_index=True)
    return error_scores

#######################################

#### MAIN########################

################################
link = 'paldal_ward_field.csv'
data = loadData(link)

### resample data to hour frequence (to reduce data length)
data1 = data.resample('15min').mean().reindex(pd.date_range(data.index[0],data.index[-1],freq='H'))
## fill missing data with mean value
data1 = data1.fillna(data1['Power'].mean())

# --- split data1 to "train/test" --- #
train_data, test_data = splitData(data=data1, trainPercent=0.7, split_by_time=False)



# --- scaling --- #
train_data_sc, scaler = scale_to_0and1(train_data)
test_data_sc = scaler.transform(test_data)
# --- create data1set with window size --- #
inputTrain, ouputTrain = \
    sequentialize(train_data_sc, train_data.index, window_size=168, to_ndarray=True)
inputTest, ouputTest = \
    sequentialize(test_data_sc, test_data.index, window_size=168, to_ndarray=True)

# # --- create data1 matrix ---#
# # todo: understand "create_data1set" function
# trainX, trainY = create_data1set(train_data1, look_back=1)
# testX, testY = create_data1set(test_data1, look_back=1)

# --- change input (X) format for LSTM  (reshape)--- #
### only with input data, output does not require reshape
inputTrain = reshapeForLSTM(inputTrain, time_steps=168)
inputTest = reshapeForLSTM(inputTest, time_steps=168)

# ###########tuning paramters
# epochs = [500, 1000, 1500, 2000, 2500, 3000]
# batch_size = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
# error_scores = optimizing(X_train, y_train, X_test, y_test,epochs,batch_size)
# print(min(error_scores['RMSE']))
########### --- train LSTM network --- #
# todo: what "K.clear_session()" do...?
K.clear_session()
model = Sequential()
model.add(LSTM(units=200, input_shape=(168, 1))) # (timestep, feature)
model.add(Dense(units=1)) # output = 1
model.compile(loss="mean_squared_error", optimizer="adam")
model.summary()
# todo: inspect "EarlyStopping function()"
early_stop = EarlyStopping(monitor="loss", patience=5, verbose=1)

model.fit(inputTrain, ouputTrain, epochs=200, batch_size=168, verbose=2, callbacks=[early_stop])
###load model from file
model.save('LSTM_keras_BatchSize_168_HiddenSize_200.h5') ## save checkpoint

from keras.models import load_model
model = load_model('LSTM_keras_BatchSize_168_HiddenSize_200.h5') ### load checkpoint


# --- predict LSTM network --- #
# todo: fix codes below...!
trainPred = model.predict(inputTrain)
trainPred = scaler.inverse_transform(trainPred)
trainY = scaler.inverse_transform(ouputTrain)

testPred = model.predict(inputTest)
testPred = scaler.inverse_transform(testPred)
testY = scaler.inverse_transform(ouputTest)


# --- MSE --- #
trainScore = math.sqrt(mean_squared_error(trainY, trainPred))
testScore = math.sqrt(mean_squared_error(testY, testPred))
print("\n",
      "Train Score:   %.1f RMSE" % (trainScore), "\n",
      " Test Score:   %.1f RMSE" % (testScore))

### create a list to hold predicted values
forecastArray = []
### define forecast length
forecastLength = 240 ## 10 days
#### initialize input window
window = train_data_sc
for i in range(forecastLength):
    #### sequentializing input data
    inputTrain, outputTrain = sequentialize(window, data1.index[:len(window)], window_size=168,to_ndarray=True)
    #### reshape input due to LSTM requirement
    inputTrain = reshapeForLSTM(inputTrain, time_steps=168)
    ## predict output
    predictedSequence = model.predict(inputTrain)
    ### reshape input data and put the last predicted value into
    m = np.reshape(window, [len(window)])
    m = np.append(m, predictedSequence[-1,0])
    ### also, put last predicted value into list
    forecastArray.append(predictedSequence[-1,0])
    print('Step: {} - predicted: {}'.format(i,predictedSequence[-1, 0]))
    #### reshape input data for sequentializing
    window = np.reshape(m, [len(m), 1])

###temporally saving predicted values (because it's take time to run prediction)
forecastIndex = data1.index[len(train_data):len(train_data)+forecastLength]
### convert forecast value into dataframe
df = pd.DataFrame(forecastArray)
df = scaler.inverse_transform(df)
    ### add index to forecasted data
df = pd.DataFrame(df, index=forecastIndex)
df.to_csv('PredictedPower_240Hour.csv')
#### plotting
plt.plot(df, label='Predicted')
plt.plot(test_data, label='Real')
plt.legend()
plt.close(all)