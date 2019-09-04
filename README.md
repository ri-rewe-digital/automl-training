# Training Custom Models with AutoML Datasets #
Managing your dataset is an essential task for every machine learning project. This is especially challenging for 
multimedia datasets. They require a large storage capacity, cause high data traffic and currently do not have a
standardized management toolchain.

Especially when multiple people work on a dataset it becomes more challenging to keep track of it. Recently more cloud-based
tools like Google's *Data Labeling* get released to support researchers and engineers with their datasets.

However, on GCP there is currently no dedicated tool to manage pre-labeled datasets. While AutoML provides that
functionality, it is not the main purpose. This project gives a brief idea how you can access datasets that are managed
with AutoML and train your own custom models. 

### Setup ###
The training is done with *TensorFlow* and *Keras*. Pandas handles the csv import. All that is left, are GCP libraries.
```bash
pip3 install -r requirements.txt
```

### GCP Service Account ###
You need to create a service account that has read access on AutoML (to access the dataset and create the export operation)
and read/write access on Storage (to export the dataset csv to the bucket and download it). Export the json-keyfile to your
machine, since it is required to run this code.

### Run ###
You can run with default
```bash
GOOGLE_APPLICATION_CREDENTIALS= <json keyfile> python3 run.py --project <project id> --dataset <dataset icn> --bucket <automl bucket name>
```
To change the training duration you can additionally set the *epochs* and *batch_size* parameters. Keep in mind that the dataset
is streamed without caching, what slows down the training. Please refer to the medium article below for details.

### References ###
The full explanation you can find [here](https://medium.com/ri-rewe-digital/manage-your-image-dataset-with-automl)