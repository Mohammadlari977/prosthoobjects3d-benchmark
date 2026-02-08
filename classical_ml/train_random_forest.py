import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

DATA_FOLDER = '../processed_data'
MODEL_SAVE_PATH = '../models/Model_RF.joblib'

def train_rf():
    print("Loading data for Random Forest (RF)...")
    X = np.load(os.path.join(DATA_FOLDER, 'points.npy'))
    y = np.load(os.path.join(DATA_FOLDER, 'labels.npy'))
    
    # Flatten Data (N, 2048, 3) -> (N, 6144)
    print("Flattening 3D points to 1D vectors...")
    X_flat = X.reshape(X.shape[0], -1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)
    
    print("Training RF Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(f"RF Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    
    if not os.path.exists('../models'): os.makedirs('../models')
    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"RF Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_rf()
