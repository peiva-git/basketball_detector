from classification.DatasetBuilder import DatasetBuilder

builder = DatasetBuilder('/mnt/DATA/tesi/dataset/dataset_classification/pallacanestro_trieste')
builder.configure_datasets_for_performance()
train_dataset, validation_dataset = builder.build()



