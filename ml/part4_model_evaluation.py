import os
import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_FOLDER = 'ready_for_train'
MODEL_PATH = 'models/Model_PointNet.keras'

def evaluate_model():
    print("Loading test data and model...")
    X = np.load(os.path.join(DATA_FOLDER, 'X_norm.npy'))
    y_cat = np.load(os.path.join(DATA_FOLDER, 'y_cat.npy'))
    y_true = np.argmax(y_cat, axis=1)

    model = keras.models.load_model(MODEL_PATH)
    
    print("Running inference...")
    y_pred_prob = model.predict(X)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("\n--- PointNet Evaluation Report ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate_model()
