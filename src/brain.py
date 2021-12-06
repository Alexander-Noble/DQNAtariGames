from keras import layers
from keras import models
import numpy as np

class Brain:
    def __init__(self, actionCount, stateCount):
        self.actionCount = actionCount
        self.stateCount = stateCount
        self.model = self.model()

    def model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (8,8), activation='relu'))
        model.add(layers.MaxPooling2D((4,4)))
        model.add(layers.Conv2D(64, (8,8), activation='relu'))
        model.add(layers.MaxPooling2D((4,4)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.actionCount))
        model.compile(loss='mse', optimizer='RMSprop')

        return model

    def predict(self, s):
        return self.model.predict(s)

    def predictOneStep(self, s):
        return self.model.predict(s.reshape(1, self.stateCount, 160, 3))

    def train(self, x, y, batch, EPOCHS):
        self.model.fit(x, y, batch_size=batch, epochs=EPOCHS)
