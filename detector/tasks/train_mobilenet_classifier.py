import os

from detector.models.classification import MobileNet
from detector.models import get_classification_model_callbacks
from detector.dataset_builders import ClassificationDatasetBuilder

import tensorflow as tf

if __name__ == '__main__':
    builder = ClassificationDatasetBuilder('/mnt/DATA/tesi/dataset/dataset_classification/pallacanestro_trieste/')
    builder.configure_datasets_for_performance(shuffle_buffer_size=20000)
    train_dataset, val_dataset = builder.train_dataset, builder.validation_dataset

    detector = MobileNet()
    detector.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    detector.model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        callbacks=get_classification_model_callbacks(detector.model_name, early_stop_patience=10, reduce_lr_patience=5)
    )
    detector.model.save(filepath=os.path.join('out', 'models', 'TF', detector.model_name), save_format='tf')
    detector.model.save(filepath=os.path.join('out', 'models', 'HDF5', detector.model_name + '.h5'), save_format='h5')
    detector.model.save(filepath=os.path.join('out', 'models', 'Keras_v3', detector.model_name + '.keras'))
