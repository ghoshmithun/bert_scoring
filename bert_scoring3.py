from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
import torch
import numpy as np
import pandas as pd
import argparse
import warnings
warnings.filterwarnings("ignore")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
configuration = model.config
configuration.output_hidden_states=True

def make_embeddings(text):
    text = text.lower()
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embed=outputs[-1][0].detach().numpy()
    out_last = np.mean(embed,axis=0)
    return out_last[0]

def create_embedding(input_path,output_path):
	data=pd.read_csv(input_path,header=None)
	data['embedding']=data.iloc[:,0].apply(make_embeddings)
	data.to_csv(output_path)


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description='Embedding creation for queries')
    argument_parser.add_argument('--path_input', type=str,default="sample_query.txt", help='Path Name', required=False)
    argument_parser.add_argument('--path_output', type=str,default="query_output.csv", help='Path Name', required=False)
    arguments = argument_parser.parse_args()
    input_path = arguments.path_input
    output_path = arguments.path_output
    create_embedding(input_path,output_path)

