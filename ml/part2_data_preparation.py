import os
import numpy as np
from tensorflow.keras.utils import to_categorical

# Configuration
DATA_FOLDER = 'processed_data'
OUTPUT_FOLDER = 'ready_for_train'

def prepare_data():
    print("Loading extracted features...")
    X = np.load(os.path.join(DATA_FOLDER, 'points.npy'))
    y = np.load(os.path.join(DATA_FOLDER, 'labels.npy'))
    
    # Normalize to unit sphere
    print("Normalizing point clouds...")
    X -= np.mean(X, axis=1, keepdims=True)
    X /= np.max(np.linalg.norm(X, axis=2, keepdims=True), axis=1, keepdims=True)

    # One-hot encoding for Deep Learning
    num_classes = len(np.unique(y))
    y_cat = to_categorical(y, num_classes)

    # Save prepared data
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    np.save(os.path.join(OUTPUT_FOLDER, 'X_norm.npy'), X)
    np.save(os.path.join(OUTPUT_FOLDER, 'y_cat.npy'), y_cat)
    print(f"Data normalization complete. Saved to {OUTPUT_FOLDER}")

if __name__ == "__main__":
    prepare_data()
