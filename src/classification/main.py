from dataset_builder import DatasetBuilder
from classification.models import SimpleClassifier
import tensorflow as tf
import matplotlib.pyplot as plt

builder = DatasetBuilder('assets/test-sample-data')
builder.configure_datasets_for_performance()
train_dataset, validation_dataset = builder.build()

model = SimpleClassifier()
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

model.save(filepath='out/models/simple-classifier', save_format='tf')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
