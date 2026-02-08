import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configuration
DATA_FOLDER = 'ready_for_train'
MODEL_SAVE_PATH = 'models/Model_PointNet.keras'
BATCH_SIZE = 32
EPOCHS = 30

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)

def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)

def tnet(inputs, num_features):
    x = conv_bn(inputs, 64)
    x = conv_bn(x, 128)
    x = conv_bn(x, 1024)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 512)
    x = dense_bn(x, 256)
    x = layers.Dense(num_features * num_features, kernel_initializer="zeros", bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()), activity_regularizer=keras.regularizers.L2(0.001))(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

def get_pointnet_model(num_points, num_classes):
    inputs = keras.Input(shape=(num_points, 3))
    x = tnet(inputs, 3)
    x = conv_bn(x, 64)
    x = conv_bn(x, 64)
    x = tnet(x, 64)
    x = conv_bn(x, 64)
    x = conv_bn(x, 128)
    x = conv_bn(x, 1024)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 512)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="pointnet")

if __name__ == "__main__":
    if not os.path.exists('models'): os.makedirs('models')

    X = np.load(os.path.join(DATA_FOLDER, 'X_norm.npy'))
    y = np.load(os.path.join(DATA_FOLDER, 'y_cat.npy'))
    
    print("Initializing PointNet model...")
    model = get_pointnet_model(X.shape[1], y.shape[1])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    print("Starting training on GPU...")
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)
    model.save(MODEL_SAVE_PATH)
    print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")
