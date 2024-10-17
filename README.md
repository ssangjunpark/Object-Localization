# Object Localization

The following project is implemented with Tensorflow's Keras API (tf.\__version__ == '2.13.0'). 
The model is trained on the fruits-360 and custom background dataset. 

<br/>

The goal of this project is to predict accurate ground truth of objects (fruits-360 dataset) using custom VGG16 classifier. util.py includes methods for Custom Data Agumentation, synthetic data generation, and custom VGG16 with custom loss functions. 

<br/>

![ol1](https://github.com/user-attachments/assets/366c472d-81af-4e8e-8db8-4604938f17ad)
![ol2](https://github.com/user-attachments/assets/79bad2a8-c274-462c-841d-37f55ceeebd1)
![ol5](https://github.com/user-attachments/assets/bc2ae60f-6037-46ca-bb19-6eda0b813047)
![ol3](https://github.com/user-attachments/assets/d12e74e0-07aa-49ed-bc77-5ed493939d22)

<br/>

Results show that the model accurately predicts the class the fruit belongs to; however, it sometimes have difficulty in prediciting accurate ground truth.

## Test.py

Pretrained model can be found here: https://drive.google.com/file/d/1o9hRzxowxjTZPb4agXqLClJ1pvkzzdiT/view?usp=sharing

In order to use the model, create a directory 'call' and save the pretrained model in the .h5 format. Execute 'Test.py'.


## Train.py
In order to train the model, execute Train.py, with desired classes (line 6), path to desired background image data in .jp*g format (line 9), and path to desired image data in .png format(line 10). 

Once the model has been trained on custom data, the classes atrribute on Test.py, ConverToPNG.py and util.py must be edited. 

## ConvertToPNG.py
ConvertToPNG.py can be used to generate .PNG with .jp*g image data. Boundaries for setting alpha channel as 0 can be found on line 26 and 27 on util.py

## util.py
Contains all utilities for Test.py, Train.py, and ConvertToPNG.py

