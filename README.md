## About the project
This is a toy project completed with the aim of getting some hands-on experience with [Gradio](https://gradio.app/) and the [transformers](https://huggingface.co/docs/transformers) library. The resulting [demo](https://huggingface.co/spaces/Yozhikoff/paper-topic-classification) can perform multilabel classification of papers from arXiv (mostly ML/AI) into different topics - note that since the classification is **multilabel** the probabilities don't have to sum to 1!

## Dataset
Approximately 30k papers from this [kaggle dataset](https://www.kaggle.com/neelshah18/arxivdataset/) were used for training. 

## Model 
I fine-tuned all weights from a pre-trained [distilbert](https://huggingface.co/Wi/arxiv-topics-distilbert-base-cased) for 2 epochs. More details on training in the notebook.
