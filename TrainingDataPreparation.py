# Flow From Directory Data Generation

# Importing Keras for Image Classification
import keras
from keras.layers import Dense,Conv2D, Flatten, MaxPool2D, Dropout
from keras.models import Sequential
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow as tf

# Expand the size of dataset with new transformed images from the original dataset using ImageDataGenerator.
train_datagen = image.ImageDataGenerator(
                  featurewise_center=False,
                  rotation_range=5,
                  fill_mode='nearest',
                  horizontal_flip=True,
                  vertical_flip=True, 
                  brightness_range=[0.5,1.3], 
                  rescale = 1./255,
                  channel_shift_range = 1.0  
                  )
val_datagen = image.ImageDataGenerator(rescale = 1./255)
test_datagen = image.ImageDataGenerator(rescale = 1./255)

train_data = train_datagen.flow_from_directory(
    directory= "./train", 
    target_size=(224,224), 
    batch_size=64, 
    class_mode = 'categorical')

train_data.class_indices

val_data = val_datagen.flow_from_directory(
    directory= "./val", 
    target_size=(224,224), 
    batch_size=64, 
    class_mode = 'categorical')

test_data = test_datagen.flow_from_directory(directory= "./test", 
                                             target_size=(224, 224), 
                                             batch_size=64, 
                                            class_mode = 'categorical',
                                             shuffle=False)

drive_location='/content/drive/MyDrive/check'
