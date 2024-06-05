import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil

#----------------------------------------load and adjust pretrained model to match our input and output
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# freeze the top layer shape = (batch_size, 8, 8, 1280)
x = base_model.output
# shape = (batch_size, 1280)
x = GlobalAveragePooling2D()(x)
# shape = (batch_size, 1024)
x = Dense(1024, activation='relu')(x)

predictions = Dense(1, activation='sigmoid')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

#-------------------------------------------Data augmentation and data generator for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Reserve 20% of data for validation
)

# Training and validation data generators
your_training_validation_dataset = "training_validation/dataset"
train_generator = train_datagen.flow_from_directory(
    your_training_validation_dataset,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    your_training_validation_datase,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

#--------------------------------------------------Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

#---------------------------------------------------Test the model
your_test_dataset = "test/dataset"
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    your_test_dataset,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test accuracy: {test_acc}')

#----------------------------------------------------Test and display the result to double checking 
#--------------------(needed in the training process, as sometimes, the test data is a bit limited to the dataset and can cause overfitting
#--------------------If the double checking result is bad, but the testing result is good, try adding more various data that cannot be 
#--------------------detected by the model in the double checking, using different dataset
def load_and_preprocess_image(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to match the input size of the model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def flow():
    # Path to the image you want to predict
    img_parent_path = "parent/path"
    img_list = [os.path.join(img_parent_path, img_name) for img_name in os.listdir(img_parent_path)]

    for img_path in img_list:
      # Preprocess the image
      preprocessed_image = load_and_preprocess_image(img_path)

      # Make a prediction
      prediction = model.predict(preprocessed_image)

      # Interpret the prediction
      # If using a sigmoid activation function for binary classification, the output will be a probability
      if prediction[0][0] >= 0.5:
          print("Orangutan detected")
      else:
          print("No orangutan detected")

      # Optionally display the image with the prediction
      img = image.load_img(img_path)
      plt.imshow(img)
      plt.title(f'Prediction: {"Orangutan detected" if prediction[0][0] >= 0.5 else "No orangutan detected"}')
      plt.axis('off')
      plt.show()

flow()

#-------------------------------------if model's prediction is fine, we can then save the model
model_path = "path/to/model.h5"
model.save(model_path)

#-------------------------------------for our project, we are using Raspberry PI, thus we need a lightweight model, so we are using tf lite to deploy our model
# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model to a file
path_to_tflite_model = 'path/to/tflite_model.tflite'
with open(path_to_tflite_model, 'wb') as f:
    f.write(tflite_model)
