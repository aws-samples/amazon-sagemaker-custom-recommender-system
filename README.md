## Build a Customized Recommender System on Amazon SageMaker

## Summary 

Recommender systems have been used to tailor customer experience on online platforms. [Amazon Personalize](https://aws.amazon.com/personalize/) is a fully-managed service that makes it easy to develop recommender system solutions; it automatically examines the data, performs feature and algorithm selection, optimizes the model based on your data, and deploys and hosts the model for real-time recommendation inference. However, due to unique constraints in some domains, sometimes recommender systems need to be custom-built. 

In this project, I will walk you through how to build and deploy a customized recommender system using Neural Collaborative Filtering model in Tensorflow 2.0 on [Amazon SageMaker](https://aws.amazon.com/sagemaker/), based on which you can customize further accordingly.

## Getting Started

[Create an Amazon SageMaker notebook instance](https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-create-ws.html) (a `ml.t2.medium` instance will suffice to run the notebooks for this project)

## Running Notebooks

There are two notebooks associated with this project:  
1. [data preparation notebook.ipynb](data-preparation-notebook.ipynb)  
This notebook contains data preprocessing code. It downloads MovieLens dataset, performs training testing split and negative sampling, and uploads processed data onto Amazon S3.  
2. [model training notebook.ipynb](model-training-notebook.ipynb)  
This notebook requires [ncf.py](ncf.py) file to run. It initiates a [Tensorflow estimator](https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/sagemaker.tensorflow.html) to train the model, then deploys the model as an [endpoint](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-hosting.html) on Amazon SageMaker Hosting Services. Lastly, it shows how to make batch recommendation inference using the model endpoint.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

## Acknowledgement

MovieLens dataset provided by GroupLens.

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872
