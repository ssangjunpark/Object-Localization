from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Concatenate
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from skimage.transform import resize
from glob import glob


classes = ['Apple Golden 1','Avocado 1','Lemon 1','Mango 1','Kiwi 1','Banana 1','Strawberry 1','Raspberry 1']


def convert_image_and_save(image_path, save_path):
    name = image_path.split("/")[-1]
    img = Image.open(image_path)

    img = img.convert("RGBA")

    lower_bound = (240, 240, 240)
    upper_bound = (255, 255, 255)

    datas = img.getdata()

    new_data = []
    for item in datas:
        r, g, b, a = item
        if lower_bound[0] <= r <= upper_bound[0] and lower_bound[1] <= g <= upper_bound[1] and lower_bound[2] <= b <= upper_bound[2]:
            new_data.append((0,0,0,0))
        else:
            new_data.append(item)

    img.putdata(new_data)
    img.save(save_path + '/' + name.split('.')[0] + '.png', 'PNG')
    print(f"Image saved as {name}.png")

def get_fruits(location, classes):
    fruits = []
    for i in range(len(classes)):
        fruits.append([])

    for i in range(len(classes)):
        fruit_files = glob(location+'/'+classes[i]+'/*.png')

        for fruit in fruit_files:
            ff = np.array(imageio.imread(fruit))
            fruits[i].append([ff, ff.shape[0], ff.shape[1], ff.shape[2]])

    return fruits

def custom_loss(Y, Y_hat):
    hypBCEGroundTruth=1
    hypBCEBinaryClass=0.5
    nypCCEClasses=1
    BCEGroundTruth = binary_crossentropy(Y[:, :4], Y_hat[:, :4]) # ground truth
    BCEBinaryClass = binary_crossentropy(Y[:, -1], Y_hat[:, -1]) # check if the object exists
    CCEClasses = categorical_crossentropy(Y[:, 4:4+len(classes)], Y_hat[:, 4:4+len(classes)]) # softmax

    return hypBCEGroundTruth * BCEGroundTruth * Y[:, -1] + nypCCEClasses * CCEClasses * Y[:, -1] + hypBCEBinaryClass * BCEBinaryClass


def make_Custom_VGG16_model(classes, custom_loss_function, BACKGROUND_SIZE):
    vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=[BACKGROUND_SIZE, BACKGROUND_SIZE, 3])
    x = Flatten()(vgg.output)
    xGroundTruth = Dense(4, activation='sigmoid')(x)
    xClasses = Dense(len(classes), activation='softmax')(x)
    xBinaryClass = Dense(1, activation='sigmoid')(x)
    x = Concatenate()([xGroundTruth, xClasses, xBinaryClass])
    model = Model(vgg.input, x)

    model.compile(loss=custom_loss_function, optimizer=Adam(lr=0.0001))
    return model

def get_backgrounds(location):
    backgrounds = []

    background_files = glob(location+'/*.jpg')
    for f in background_files:
        bg = np.array(image.load_img(f))
        backgrounds.append(bg)

    return backgrounds

def data_generator(fruits, backgrounds, BACKGROUND_SIZE, IMAGE_SIZE, batch_size=64, num_batches=50, willFlip=True, willResize=True, appear_cap = 0.5):
    while True:

        for _ in range(num_batches):
            X = np.zeros((batch_size, BACKGROUND_SIZE, BACKGROUND_SIZE, 3))
            Y = np.zeros((batch_size, 5+len(classes)))

            for i in range(batch_size):
                bg_id = np.random.choice(len(backgrounds))
                background = backgrounds[bg_id]
                background_H, background_W, background_C = background.shape

                random_background_row = np.random.randint(background_H - BACKGROUND_SIZE)
                random_background_column = np.random.randint(background_W - BACKGROUND_SIZE)
                
                X[i] = background[random_background_row:random_background_row+BACKGROUND_SIZE, random_background_column:random_background_column+BACKGROUND_SIZE].copy()

                # calculate the chance of fruit appearing
                appear_prob = np.random.random()

                if appear_prob > appear_cap:

                    # choose random fruit
                    class_id = np.random.randint(low=0, high=len(classes))
                    image_id = np.random.randint(low=0, high=len(fruits[class_id]))
                    fruit, fruit_H, fruit_W, fruit_C = fruits[class_id][image_id]

                    #resize and convert
                    if willResize:
                        scaleFactor = np.random.random() + 0.5 # 0.5 ~ 1.5
                        fruit_new_H = int(fruit_H * scaleFactor)
                        fruit_new_W = int(fruit_W * scaleFactor)
                        transfomredFruit = resize(
                            fruit,
                            (fruit_new_H, fruit_new_W),
                            preserve_range=True
                        ).astype(np.uint8)
                    else:
                        transfomredFruit = resize(
                            fruit,
                            (fruit_H, fruit_W),
                            preserve_range=True
                        ).astype(np.uint8)

                    if willFlip:
                        if np.random.random() > 0.5:
                            transfomredFruit = np.fliplr(transfomredFruit)

                    row0 = np.random.randint(BACKGROUND_SIZE - fruit_new_H)
                    col0 = np.random.randint(BACKGROUND_SIZE - fruit_new_W)
                    row1 = row0 + fruit_new_H
                    col1 = col0 + fruit_new_W


                    # layer
                    mask = (transfomredFruit[:,:,3] == 0)
                    backgroundSlice = X[i,row0:row1,col0:col1,:]
                    backgroundSlice = np.expand_dims(mask, -1) * backgroundSlice
                    backgroundSlice += transfomredFruit[:,:,:3]
                    X[i,row0:row1,col0:col1, :] = backgroundSlice

                    Y[i, 0] = row0/BACKGROUND_SIZE
                    Y[i, 1] = col0/BACKGROUND_SIZE
                    Y[i, 2] = row1/BACKGROUND_SIZE
                    Y[i, 3] = col1/BACKGROUND_SIZE

                    Y[i, 4+class_id] = 1

                Y[i, 4+len(classes)] = (appear_prob > appear_cap)
        yield X / 255.0, Y


def draw_prediction(x,p, BACKGROUND_SIZE):
    fig, ax = plt.subplots(1)
    ax.imshow(x[0])
    if p[0][-1] > 0.5:
        predicted_class_id = np.argmax(p[0][4:4+len(classes)])
        predicted_class = classes[predicted_class_id]
        print(predicted_class)
        rect = Rectangle( (p[0][1]*BACKGROUND_SIZE, p[0][0]*BACKGROUND_SIZE), p[0][3]*BACKGROUND_SIZE, p[0][2]*BACKGROUND_SIZE,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    else:
        print("No classes!")
    
    plt.show()