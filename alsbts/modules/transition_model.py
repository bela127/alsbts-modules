from keras.models import Sequential
from keras.layers import Dense

class LSTMTransitionModel():
    

    def __init__(self) -> None:
        # design the neural network model
        self.model = Sequential()
        self.model.add(
            Dense(10, input_dim=1, activation="relu", kernel_initializer="he_uniform")
        )
        self.model.add(Dense(10, activation="relu", kernel_initializer="he_uniform"))
        self.model.add(Dense(1))
        # define the loss function and optimization algorithm
        self.model.compile(loss="mse", optimizer="adam")