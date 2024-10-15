from util import custom_loss, get_backgrounds, get_fruits, make_Custom_VGG16_model, data_generator

BACKGROUND_SIZE = 200
IMAGE_SIZE = 100

classes = ['Apple Golden 1','Avocado 1','Lemon 1','Mango 1','Kiwi 1','Banana 1','Strawberry 1','Raspberry 1']
num_classes = len(classes)

backgrounds = get_backgrounds(location='backgrounds')
fruits = get_fruits(location='fruits-360-transparent/Test', classes=classes)

model = make_Custom_VGG16_model(classes=classes, custom_loss_function=custom_loss, BACKGROUND_SIZE=BACKGROUND_SIZE)
model.summary()

model.fit_generator(
    data_generator(fruits=fruits, backgrounds=backgrounds, BACKGROUND_SIZE=BACKGROUND_SIZE, IMAGE_SIZE=IMAGE_SIZE, batch_size=64, num_batches=50, willFlip=True, willResize=True, appear_cap = 0.5),
    steps_per_epoch=50,
    epochs=5,
)

model.save('call/test.h5')