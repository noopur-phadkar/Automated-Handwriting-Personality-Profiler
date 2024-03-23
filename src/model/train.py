from src.model.model import create_cnn_model
from sklearn.model_selection import train_test_split
from keras.models import load_model
import numpy as np
import os

def train_model(X_train, y_train, X_test, y_test):
    # Define the input shape and number of classes based on the training data
    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1]

    # Create the CNN model
    model = create_cnn_model(input_shape, num_classes)

    # Split the training data for validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Train the model
    history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))

    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.3f}, Test loss: {test_loss:.3f}")

    # Save the model
    model.save('handwriting_personality_model.h5')

    return history

# Function to load the saved model
def load_saved_model(model_path):
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        print("Model file not found!")
        return None

# Optional: Function to continue training from a saved model
def continue_training(model_path, X_train, y_train, X_test, y_test):
    model = load_saved_model(model_path)
    if model:
        history = model.fit(X_train, y_train, batch_size=32, epochs=5, validation_split=0.2)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
        print(f"Test accuracy after continued training: {test_acc:.3f}, Test loss: {test_loss:.3f}")
        return history
    return None
