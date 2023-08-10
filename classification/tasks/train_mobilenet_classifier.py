import os

from classification.models.classification import MobileNet, get_model_callbacks
from classification.dataset_builders import ClassificationDatasetBuilder

import tensorflow as tf

if __name__ == '__main__':
    builder = ClassificationDatasetBuilder('/mnt/DATA/tesi/dataset/dataset_classification/pallacanestro_trieste/')
    builder.configure_datasets_for_performance(shuffle_buffer_size=20000)
    train_dataset, val_dataset = builder.train_dataset, builder.validation_dataset

    model = MobileNet()
    model.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    model.model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        callbacks=get_model_callbacks(model.model_name, early_stop_patience=10, reduce_lr_patience=5)
    )
    model.model.save(filepath=os.path.join('out', 'models', 'TF', model.model_name), save_format='tf')
    model.model.save(filepath=os.path.join('out', 'models', 'HDF5', model.model_name + '.h5'), save_format='h5')
    model.model.save(filepath=os.path.join('out', 'models', 'Keras_v3', model.model_name + '.keras'))
