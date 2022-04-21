import os
# Disable TF messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import cv2  # OpenCV for image manipulation; pip install opencv-python

# Directory with test images
TEST_DATASET = 'mnist-test'

# Trained model name
MODEL = 'model.h5'

# Load trained model of neural network
model = tf.keras.models.load_model(MODEL)

for image_name in os.listdir(TEST_DATASET):
    
    # Load the image
    image = cv2.imread(TEST_DATASET + os.path.sep + image_name,
        cv2.IMREAD_GRAYSCALE)
        
    # Save original for further displaying
    orig_image = image.copy()
    
    # Pre-process the image for classification
    image = image.astype('float32') / 255
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    # Classify the input image
    prediction = model.predict(image)[0]
    
    # Winner class as index of maximum value in prediction
    winner_class = np.argmax(prediction)
        
    # Winner probability as maximum value in prediction * 100
    winner_probability = round(np.max(prediction)*100,2)
    
    # Build the text label
    label = "Recognized {} with probability {}%".format(winner_class, 
                                                winner_probability)
    
    # Draw the label on the image
    output_image = cv2.resize(orig_image, (600,600))
    cv2.putText(output_image, label, (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)

    # Show the output image        
    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
