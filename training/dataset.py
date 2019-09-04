import logging
import os

import pandas
from google.cloud import storage
from google.cloud.automl_v1beta1 import AutoMlClient

from training.input_fn import input_fn

compute_region = 'us-central1'  # only supported region in beta
dataset_file_name = 'dataset.csv'


def export_dataset(project: str,
                   dataset: str,
                   bucket: str):
    if os.path.isfile(dataset_file_name):
        logging.info('Dataset already downloaded, no export done.')
        return dataset_file_name
    client = AutoMlClient()
    export_path = 'gs://{}/export/export_{}'.format(bucket, dataset)
    output_config = {"gcs_destination": {"output_uri_prefix": export_path}}
    dataset_name = client.dataset_path(project,
                                       compute_region,
                                       dataset)
    export_operation = client.export_data(dataset_name, output_config)
    logging.info('Waiting for the export to complete...')
    export_operation.result()
    logging.info('Downloading exported csv...')
    download_training_csv(bucket, 'export/export_{}/export.csv'.format(dataset), dataset_file_name)
    return dataset_file_name


def download_training_csv(bucket_name: str,
                          source_blob_name: str,
                          destination_file_name: str):
    # https://cloud.google.com/storage/docs/downloading-objects#storage-download-object-python
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    return destination_file_name


def load_data_array(csv_file: str):
    df = pandas.read_csv(csv_file, names=['SET', 'FILE', 'LABEL'])

    # get the labels as string
    labels = df['LABEL'].unique()
    logging.debug('labels: {}'.format(labels))

    # convert the label from string to categorical numerics
    df['LABEL'] = df['LABEL'].astype('category').cat.codes

    return df, list(labels)


def split_dataset(df, dataset:str = 'TRAIN'):
    assert dataset in ['TRAIN', 'VALIDATION', 'TEST'], 'Not a valid dataset'
    sub_df = df[df['SET'] == dataset]
    logging.debug("Dataframe header: \n{}".format(sub_df.head()))
    return sub_df


def create_iterator(df, num_classes: int, epochs: int, batch_size: int, is_training: bool = False):
    images, labels, init_op = input_fn(filenames=df['FILE'].to_list(),
                                       labels=df['LABEL'].to_list(),
                                       num_classes=num_classes,
                                       batch_size=batch_size,
                                       epochs=epochs,
                                       is_training=is_training)
    return images, labels
