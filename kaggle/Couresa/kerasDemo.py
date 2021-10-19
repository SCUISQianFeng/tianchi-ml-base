
from tensorflow import keras
from tensorflow.keras import layers
input_shape = [11]



model = keras.models.Sequential()
model.add(keras.Input(shape=input_shape))
model.add(layers.Dense(units=1))

model.weights

w, b = model.weights, model.biases
print("Weights\n{}\n\nBias\n{}".format(w, b))