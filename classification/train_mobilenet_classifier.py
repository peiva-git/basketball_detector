import os

import classification
from classification.models.models import MobileNet, get_model_callbacks

import tensorflow as tf

if __name__ == '__main__':
    builder = classification.ClassificationDatasetBuilder('/home/ubuntu/classification_dataset/pallacanestro_trieste')
    builder.configure_datasets_for_performance(shuffle_buffer_size=500000)
    train_dataset, val_dataset = builder.train_dataset, builder.validation_dataset

    model = MobileNet()
    model.model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=tf.keras.metrics.Accuracy()
    )

    model.model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        callbacks=get_model_callbacks(model.model_name, early_stop_patience=10, reduce_lr_patience=8)
    )
    model.model.save(filepath=os.path.join('out', 'models', 'TF', model.model_name), save_format='tf')
    model.model.save(filepath=os.path.join('out', 'models', 'HDF5', model.model_name + '.h5'), save_format='h5')

