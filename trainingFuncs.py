print("Loading imports...")

from os import listdir
import tensorflow as tf
import numpy as np

print("Imports loaded.")

AnnotatedData = tuple[np.ndarray, np.ndarray] # (data, annotations)
ModelData = tuple[AnnotatedData, AnnotatedData] # one for training, one for testing


def makeSequentialModel(layers : list, compile_ : bool = True, learningRate : float = 0.0003) -> tf.keras.models.Sequential:
    """ Creates a sequential model from a list of layers.
    
    Attributes
    ----------
    layers : list
        A list of layers to add to the model.
        
    Returns
    -------
    model : tensorflow.keras.models.Sequential
        The sequential model.
    
    """
    model = tf.keras.models.Sequential()
    for layer in layers:
        model.add(layer)
    if compile_:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def getTrainTestData(annotatedData : list[AnnotatedData], trainSplit : float = 0.8, equalPositiveAndNegative : bool = True) -> ModelData:
    """ Gets the training and test data from a list of annotated data.

    Parameters
    ----------
    modelType : ModelType
        The type of model to train.
    annotatedData : list[AnnotatedData]
        The annotated data to use.
    shuffle : bool
        Whether to shuffle the data before training.
    equalPositiveAndNegative : bool
        Whether to equalize the number of positive and negative samples before training.

    Returns
    -------
    ModelData
        The training and test data, as a tuple of ((trainX, trainY), (testX, testY)).
    """
    
    combinedInputs = np.concatenate(list(map(lambda x: x[0], annotatedData))), np.concatenate(list(map(lambda x: x[1], annotatedData)))

    # split the data into training and test sets (the model data); (trainSplit * 100%) of the data is used for training
    trainLength = int(len(combinedInputs[0]) * trainSplit)
    modelData = (combinedInputs[0][:trainLength], combinedInputs[1][:trainLength]), (combinedInputs[0][trainLength:], combinedInputs[1][trainLength:])
    return _equalisePositiveAndNegative(modelData, shuffle=True)


def _equalisePositiveAndNegative(combined : ModelData, shuffle : bool) -> ModelData:
    """ Equalises the number of positive and negative examples in both the training and test sets (individually). """
    train, test = combined
    return _equalisePNForSingleSet(train, shuffle), _equalisePNForSingleSet(test, shuffle)


def _equalisePNForSingleSet(annotatedData : AnnotatedData, shuffle : bool) -> AnnotatedData:
    data, annotations = annotatedData
    positiveIndices = np.where(annotations == 1)[0]
    negativeIndices = np.where(annotations == 0)[0]
    
    np.random.shuffle(positiveIndices) # shuffle the indices so we don't always remove the last ones
    np.random.shuffle(negativeIndices)
    
    if len(positiveIndices) > len(negativeIndices):
        positiveIndices = positiveIndices[:len(negativeIndices)]
    elif len(negativeIndices) > len(positiveIndices):
        negativeIndices = negativeIndices[:len(positiveIndices)]
    
    indices = np.concatenate((positiveIndices, negativeIndices))
    if not shuffle:
        # if we're not going to shuffle later, need to sort the indices back into the original order
        indices = np.sort(indices)
    
    return data[indices], annotations[indices]


def trainModel(model : tf.keras.models.Sequential, data : ModelData, epochs : int, batchSize : int, fracVal : float = 0.1, saveCheckpoints : bool = True, resampleFrequency : int = -1):
    """ Trains a model on the data in a given directory.
    
    Attributes
    ----------
    modelType : ModelType
        The type of model to train.
    model : tensorflow.keras.models.Sequential
        The tensorflow model on which to train.
    data : ModelData
        The annotated data to use, as a tuple of ((trainX, trainY), (testX, testY)).
    epochs : int
        The number of epochs to train for.
    batchSize : int
        The batch size to train with.
    fracVal : float
        The fraction of the training data to use for validation.
    saveCheckpoints : bool
        Whether to save checkpoints during training.
    resampleFrequency : int
        The frequency at which to resample the data. If -1 (default), no resampling is done.
        
    Returns
    -------
    model : tensorflow.keras.models.X
        The trained model.
    history : tensorflow.python.keras.callbacks.History
        The training history.
    """
        
    (trainX, trainY), (testX, testY) = data
    
    valX = trainX[-round(len(trainX)*fracVal):]
    valY = trainY[-round(len(trainY)*fracVal):]
    trainX = trainX[:-round(len(trainX)*fracVal)]
    trainY = trainY[:-round(len(trainY)*fracVal)]
    
    print(f"TrainX data shape: {trainX.shape}")
    print(f"TrainY data shape: {trainY.shape}")
    print(f"ValX data shape: {valX.shape}")
    print(f"ValY data shape: {valY.shape}")
    print(f"TestX data shape: {testX.shape}")
    print(f"TestY data shape: {testY.shape}")
    
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batchSize, validation_data=(valX, valY), shuffle=True)
    
    model.summary()
    model.save(path := "model.h5")
    print(f"Model saved: {path}")
    
    model.evaluate(testX, testY)
    return model, history
