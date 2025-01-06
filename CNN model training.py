#%%
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
import pickle
import os


#%% Training data
train_data = tf.keras.utils.image_dataset_from_directory(
    'E:/New Plant Diseases Dataset(Augmented)/train',
    labels='inferred',  # takes the labels from the folder names.
    label_mode='categorical',  # converts the labels for multi-class classification into categorical
    image_size=(256, 256),
    batch_size=32
)

# Testing data
test_data = tf.keras.utils.image_dataset_from_directory(
    'E:/New Plant Diseases Dataset(Augmented)/test',
    labels='inferred',
    label_mode='categorical',
    image_size=(256, 256),
    batch_size=32
)

# Normalization
train_data = train_data.map(lambda x, y: (x / 255.0, y))
test_data = test_data.map(lambda x, y: (x / 255.0, y))

# Save the class names
class_names = train_data.class_names
num_classes = len(class_names)
print(f"Sınıf sayısı: {num_classes}")

#%% Model Definition
model = Sequential([
    Input(shape=(256, 256, 3)),  # Input Size (256x256x3)
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # The output is adjusted according to the number of classes
])
#%% Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#%% # Define a callback to save the best model

checkpoint = ModelCheckpoint(
    'best_model.keras',  # The file name where the best model will be saved
    save_best_only=True,  # Saves only the model with the best validation loss
    monitor='val_loss',  # Monitors the validation loss
    mode='min',  # Aims for the lowest validation loss
    verbose=1  # Displays the saving process in the terminal
)
#%% 
#Define a callback for early stopping

early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitors validation loss for early stopping
    patience=3,  # Number of epochs to wait for improvement
    mode='min',  # Aims for the lowest validation loss
    verbose=1,  # Displays the early stopping process in the terminal
    restore_best_weights=True  # Restores the best weights after stopping
    )


#%% Start training

history = model.fit(
    train_data,
    epochs=25,
    validation_data=test_data,
    callbacks=[checkpoint],  # Adding the callback here
    verbose=1
)

#%% Evaluate the Model
results = model.evaluate(test_data)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

#%% Load the Best Model and Evaluate
best_model = load_model('best_model.keras')
best_results = best_model.evaluate(test_data)
print(f"Best Model - Test Loss: {best_results[0]}, Test Accuracy: {best_results[1]}")
