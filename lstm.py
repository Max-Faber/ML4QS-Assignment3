import tensorflow

from definitions import ROOT_DIR
import trainTestValSplit as ttvs
import preProcessing as pp

if __name__ == "__main__":
    n_neurons = 128
    window_size = 8
    epochs = 100
    n_y = 6
    batch_size = 256
    n_features = 278

    train_concat, test_concat, val_concat = ttvs.getTrainTestValSplit(
        _dataset_path=f'{ROOT_DIR}/dataset/ExtraSensory/ExtraSensory.per_uuid_features_labels'
    )

    test_x, test_y = pp.preProcess(test_concat)
    train_x, train_y = pp.preProcess(train_concat)
    val_x, val_y = pp.preProcess(val_concat)

    model = tensorflow.keras.Sequential()
    model.add(tensorflow.keras.layers.LSTM(n_neurons, return_sequences=True, input_shape=(batch_size, n_features)))
    model.add(tensorflow.keras.layers.LSTM(n_neurons, return_sequences=True))
    model.add(tensorflow.keras.layers.LSTM(n_neurons))
    model.add(tensorflow.keras.layers.Dense(n_y, activation=tensorflow.keras.activations.softmax))
    model.compile(loss=tensorflow.keras.losses.CategoricalCrossEntropy())
    model.fit(x=train_x, y=train_y, validation_data=(val_x, val_y), epochs=epochs)
    predict = model.predict(test_x)
