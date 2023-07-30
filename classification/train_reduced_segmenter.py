import os

import keras.callbacks, keras.losses

from classification.dataset_builders import SegmentationDatasetBuilder, Basketballs
from classification.models.models import Segmentation

builder = SegmentationDatasetBuilder(data_directory='/mnt/DATA/tesi/dataset/dataset_youtube/pallacanestro_trieste'
                                                    '/stagione_2019-20_legabasket/pallacanestro_trieste'
                                                    '-ori_ora_pistoia/',
                                     masks_directory='/home/peiva/MATLABProjects/tesi/data/pallacanestro_trieste'
                                                     '/pistoia/PixelLabelData/'
                                     )
train_paths = builder.train_input_mask_path_pairs
validation_paths = builder.validation_input_mask_path_pairs

train_generator = Basketballs(train_paths, image_size=(160, 160), batch_size=1)
validation_generator = Basketballs(validation_paths, image_size=(160, 160), batch_size=1)

model = Segmentation(image_size=(160, 160)).model
model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy())

callbacks = [
    keras.callbacks.ModelCheckpoint("basketball_segmentation.h5", save_best_only=True)
]

# Train the model, doing validation at the end of each epoch.
epochs = 15
model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=callbacks)
model.save(filepath=os.path.join('out', 'models', 'TF', 'segmenter'), save_format='tf')
model.save(filepath=os.path.join('out', 'models', 'HDF5', 'segmenter' + '.h5'), save_format='h5')
