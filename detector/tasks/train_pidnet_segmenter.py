import os

from detector.dataset_builders import SegmentationDatasetBuilder
from detector.models.classification import get_model_callbacks
from detector.models.pidnet import PIDNetSmall

import tensorflow as tf

if __name__ == '__main__':
    builder = SegmentationDatasetBuilder('/mnt/DATA/tesi/dataset/dataset_segmentation/pallacanestro_trieste/')
    builder.configure_datasets_for_performance(shuffle_buffer_size=500, input_batch_size=5)
    train_dataset, validation_dataset = builder.train_dataset, builder.validation_dataset

    segmenter = PIDNetSmall(input_shape=(1024, 2048), number_of_classes=2)
    segmenter.model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(momentum=0.9, learning_rate=0.045),
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.MeanIoU(num_classes=2)]
    )
    segmenter.model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=100,
        callbacks=get_model_callbacks(segmenter.model_name, early_stop_patience=10, reduce_lr_patience=5)
    )
    segmenter.model.save(filepath=os.path.join('out', 'models', 'TF', segmenter.model_name), save_format='tf')
    segmenter.model.save(filepath=os.path.join('out', 'models', 'HDF5', segmenter.model_name + '.h5'), save_format='h5')
    segmenter.model.save(filepath=os.path.join('out', 'models', 'Keras_v3', segmenter.model_name + '.keras'))

