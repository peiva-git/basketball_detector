import os

from detector.models.classification import MobileNet
from detector.models import get_classification_model_callbacks
from detector.dataset_builders import ClassificationDatasetBuilder, ClassificationSequenceBuilder

import tensorflow as tf

if __name__ == '__main__':
    # builder = ClassificationDatasetBuilder('/home/ubuntu/classification_dataset/pallacanestro_trieste/')
    # builder.configure_datasets_for_performance(shuffle_buffer_size=20000)
    # train_dataset, val_dataset = builder.train_dataset, builder.validation_dataset
    builder = ClassificationSequenceBuilder('/home/ubuntu/classification_dataset/pallacanestro_trieste/', 8)
    train_sequence, val_sequence = builder.training_sequence, builder.validation_sequence

    classifier = MobileNet(number_of_classes=2, image_width=112, image_height=112)
    classifier.model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.metrics.Recall()
        ]
    )

    classifier.model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=100,
        verbose=1,
        callbacks=get_classification_model_callbacks(classifier.model_name, early_stop_patience=10, reduce_lr_patience=5)
    )
    classifier.model.save(filepath=os.path.join('out', 'models', 'Keras_v3', classifier.model_name + '.keras'))
    classifier.model.save(filepath=os.path.join('out', 'models', 'TF', classifier.model_name), save_format='tf')
    classifier.model.save(filepath=os.path.join('out', 'models', 'HDF5', classifier.model_name + '.h5'), save_format='h5')
