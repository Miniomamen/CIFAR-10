import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, Callback
import tensorflow.keras.backend as K

# Klasa SimpleLRScheduler
class SimpleLRScheduler(Callback):
    def __init__(self, start_lr, end_lr, num_iterations):
        super(SimpleLRScheduler, self).__init__()
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_iterations = num_iterations
        self.iteration = 0
        self.lrs = np.linspace(start_lr, end_lr, num_iterations)
        self.losses = []

    def on_batch_begin(self, batch, logs=None):
        if self.iteration < self.num_iterations:
            lr = self.lrs[self.iteration]
            K.set_value(self.model.optimizer.lr, lr)
            self.iteration += 1

    def on_batch_end(self, batch, logs=None):
        if self.iteration <= self.num_iterations:
            self.losses.append(logs['loss'])

    def on_train_end(self, logs=None):
        plt.figure(figsize=(10, 6))
        plt.plot(self.lrs[:len(self.losses)], self.losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.show()


def plot_history(history):
    plt.figure(figsize=(14, 5))

    # Wykres dokładności
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    # Wykres straty
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()


import matplotlib.pyplot as plt
import numpy as np


def display_test_predictions(model, dataset, class_names):
    plt.figure(figsize=(15, 15))
    for images, labels in dataset.take(1):  # Bierzemy jeden batch z datasetu
        predictions = model.predict(images)
        predictions_classes = np.argmax(predictions, axis=1)
        labels = np.argmax(labels, axis=1)  # Prawdziwe klasy

        # Losowanie 25 unikalnych indeksów z dostępnego zakresu
        random_indices = np.random.choice(images.shape[0], 25, replace=False)

        for i, idx in enumerate(random_indices):  # Używamy losowych indeksów
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[idx], cmap=plt.cm.binary)
            plt.xlabel(f"Actual: {class_names[labels[idx]]}\nPredicted: {class_names[predictions_classes[idx]]}")
    plt.show()
