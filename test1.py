import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('braintumor-classification.h5')

# Load and preprocess the image
img_path = 'Braintumor-multiclass\\braintumor-multiclass\\Testing\\no_tumor\\no(11).jpg'
img = cv2.imread(img_path)

# Resize to match the model input size
img = cv2.resize(img, (128, 128))
img_array = np.array(img)

# Expand dimensions to match the input shape of the model
img_array = img_array.reshape(1, 128, 128, 3)

# Predict
predictions = model.predict(img_array)
indices = predictions.argmax()

# Display the image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
plt.axis('off')  # Hide axis

# Prepare the output message
labels = ['Glioma Type of Brain Tumor', 'Meningioma Type of Brain Tumor', 'No Brain Tumor', 'Pituitary Type of Brain Tumor']
output_message = f"Prediction: {labels[indices]}"

# Show the image and prediction
plt.title(output_message)
plt.show()

