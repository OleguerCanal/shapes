from keras.models import Sequential
from keras.layers import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np

def createModel(input_shape):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(Conv2D(64, (1, 1), activation='relu'))

    model.add(Dropout(0.5))
    model.add(Conv2D(64, (1, 1), activation='linear'))

    # Compile model
    model.compile(
                loss='mean_squared_error',
                optimizer='adam')
    return model

def load_data(image_list, gradient_list):
    data = []

    shape = np.shape(image_list[0])
    xvalues = np.array(range(shape[0]))
    yvalues = np.array(range(shape[1]))

    xx, yy = np.meshgrid(xvalues, yvalues)
    pos = np.stack(xx, yy)

    for image in image_list:
        np.stack(image, pos)

def train():
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # We load the data
    data, labels = load_data

    # Data split
    (trainX, testX, trainY, testY) = train_test_split(data,
    	labels, test_size=0.25, random_state=42)

    imput_shape = np.shape(trainX[0])
    model = createModel(input_shape=input_shape)

    # convert the labels from integers to vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)




    # evaluate model with standardized dataset
    estimator = KerasRegressor(build_fn=model, epochs=100, batch_size=5, verbose=0)
