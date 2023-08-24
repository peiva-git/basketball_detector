import os

from detector.dataset_builders import SegmentationDatasetBuilder
from detector.models import get_segmentation_model_callbacks

import tensorflow as tf

from detector.models.segmentation import UNet

if __name__ == '__main__':
    builder = SegmentationDatasetBuilder('/mnt/DATA/tesi/dataset/dataset_segmentation/pallacanestro_trieste/')
    builder.configure_datasets_for_performance(shuffle_buffer_size=10, input_batch_size=2)
    train_dataset, validation_dataset = builder.train_dataset, builder.validation_dataset

    segmenter = UNet(input_shape=(1024, 2048, 3), number_of_classes=2)
    segmenter.model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(momentum=0.3, learning_rate=0.1),
        metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])]
    )
    segmenter.model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=100,
        callbacks=get_segmentation_model_callbacks(segmenter.model_name, early_stop_patience=12, reduce_lr_patience=3)
    )
    segmenter.model.save(filepath=os.path.join('out', 'models', 'Keras_v3', segmenter.model_name + '.keras'))
    segmenter.model.save(filepath=os.path.join('out', 'models', 'TF', segmenter.model_name), save_format='tf')
    segmenter.model.save(filepath=os.path.join('out', 'models', 'HDF5', segmenter.model_name + '.h5'), save_format='h5')
