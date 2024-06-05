import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import tqdm
from keras.preprocessing import image
from tflite_runtime.interpreter import Interpreter

# Load the TensorFlow Lite model
path_to_model_saved = "path/to/model.tflite"
interpreter = Interpreter(model_path=path_to_model_saved)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the image
folder_input = "input/folder"
folder_output = "output/folder"
file_paths = [os.path.join(folder_input, file_name) for file_name in os.listdir(folder_input)]

for img_path in tqdm.tqdm(file_paths):
  img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = img_array / 255.0  # Assuming the model expects scaled images

  # Set the tensor to the preprocessed image
  interpreter.set_tensor(input_details[0]['index'], img_array)

  # Run the inference
  interpreter.invoke()

  # Get the result
  output_data = interpreter.get_tensor(output_details[0]['index'])

  prediction = output_data[0][0]
  if prediction <=0.5:
    classified_as = 'no orang utan detected'
  else:
    classified_as = 'orangutan detected'

  # To show the picture
  '''print(f"Prediction: {classified_as}, File: {img_path}")
  img = image.load_img(img_path)
  plt.imshow(img)
  plt.title(f'Prediction: {classified_as}')
  plt.axis('off')
  plt.show()'''

  parent_dir, file_name = os.path.split(img_path)
  new_path = os.path.join(folder_output, classified_as+' '+file_name)
  os.rename(img_path, new_path)
