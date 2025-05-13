import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
def load_data(csv_path, img_folder):
    df = pd.read_csv(csv_path)
    images = []
    angles = []
    
    for index, row in df.iterrows():
        img_path = os.path.join(img_folder, row[0])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (200, 66))  # Resize for model
        images.append(image)
        angles.append(float(row[1]))  # Steering angle
    
    return np.array(images), np.array(angles)

# Load both datasets
X1, y1 = load_data('self_driving_car_dataset_jungle/driving_log.csv', 'self_driving_car_dataset_jungle/IMG')
X2, y2 = load_data('self_driving_car_dataset_make/driving_log.csv', 'self_driving_car_dataset_make/IMG')

# Combine datasets
X = np.concatenate((X1, X2), axis=0)
y = np.concatenate((y1, y2), axis=0)

# Save as NumPy arrays
np.save('X.npy', X)
np.save('y.npy', y)
print("Dataset preprocessing complete!")



