from util import custom_loss, get_backgrounds, get_fruits, make_Custom_VGG16_model, data_generator, draw_prediction
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BACKGROUND_SIZE = 200
IMAGE_SIZE = 100

classes = ['Apple Golden 1','Avocado 1','Lemon 1','Mango 1','Kiwi 1','Banana 1','Strawberry 1','Raspberry 1']
num_classes = len(classes)

backgrounds = get_backgrounds(location='backgrounds')
fruits = get_fruits(location='fruits-360-transparent/Test', classes=classes)

model = make_Custom_VGG16_model(classes=classes, custom_loss_function=custom_loss, BACKGROUND_SIZE=BACKGROUND_SIZE)
model.summary()

model.load_weights('call/test.h5')

for X, Y in data_generator(fruits=fruits, backgrounds=backgrounds, BACKGROUND_SIZE=BACKGROUND_SIZE, IMAGE_SIZE=IMAGE_SIZE, batch_size=1, num_batches=1, willFlip=True, willResize=True, appear_cap = 0.5):
    x, y = X, Y
    break

prediction = model.predict(x)
print("Predictions: ", prediction[0])

draw_prediction(x, prediction, BACKGROUND_SIZE)