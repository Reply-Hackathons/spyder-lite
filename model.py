import trainingFuncs

import tensorflow as tf

DATA_PATH = "" # todo

def trainLSTM(resampleFrequency: int = -1):
    
    # the data has to be collected here, not inside trainModel, as we use properties of the data (namely shape) to build the model
    # (in this model, it is only used in the optional Input layer, but other models require it for reshapes and so for consistency it is kept)
    ((trainX, trainY), (testX, testY)) = trainingFuncs.getTrainTestData(
        getData(DATA_PATH), # todo
        trainSplit=0.8, 
        equalPositiveAndNegative=True
    )
    
    return trainingFuncs.trainModel(
            trainingFuncs.makeSequentialModel([
                tf.keras.Input(shape=trainX.shape[1:], name='input'),
                tf.keras.layers.LSTM(units=128, activation='tanh', return_sequences=True),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.LSTM(units=128, activation='tanh', return_sequences=True),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.LSTM(units=128,  activation='tanh', return_sequences=False),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(units=1, activation='sigmoid')]
            ),
            ((trainX, trainY), (testX, testY)),
            fracVal=0.1,
            epochs=30, 
            batchSize=64,
            resampleFrequency=resampleFrequency
    )