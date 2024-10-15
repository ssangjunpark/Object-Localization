import os
from PIL import Image
from glob import glob 

from util import convert_image_and_save

# check if directory to save imagefile exists
directory = "fruits-360-transparent/"

if not os.path.exists(directory):
    os.makedirs(directory)

classes = [
  'Apple Golden 1',
  'Avocado 1',
  'Lemon 1',
  'Mango 1',
  'Kiwi 1',
  'Banana 1',
  'Strawberry 1',
  'Raspberry 1'
]

# location for extracting image dataset
train_path = 'fruits-360/Training'
test_path = 'fruits-360/Test'

# location to save image with .png extension
train_save_path = 'fruits-360-transparent/Training/'
test_save_path = 'fruits-360-transparent/Test/'

for cl in classes:
    # grab all image files for a specific class
    image_files = glob(train_path + '/'+cl+'/*.jp*g')
    valid_image_files = glob(test_path + '/'+cl+'/*.jp*g')

    # check if directory to save imagefile exists
    if not os.path.exists(train_save_path+cl):
        os.makedirs(train_save_path+cl)
    
    # check if directory to save imagefile exists
    if not os.path.exists(test_save_path+cl):
        os.makedirs(test_save_path+cl)

    # convert jp*g to png in image_files
    for imf in image_files:
        convert_image_and_save(imf, train_save_path+cl)

    # convert jp*g to png in valid_image_files
    for vmf in valid_image_files:
        convert_image_and_save(vmf, test_save_path+cl)