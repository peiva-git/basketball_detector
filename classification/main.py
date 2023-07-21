from .dataset_builder import DatasetBuilder
from .models import SimpleClassifier
import tensorflow as tf

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
