
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import pandas as pd
import argparse
import warnings
warnings.filterwarnings("ignore")

model = SentenceTransformer('distilroberta-base-msmarco-v1')


def create_embedding(input_path,output_path):
    data=pd.read_csv(input_path,header=None)
    sentences = data.iloc[:,0].tolist()
    sentence_embeddings = model.encode(sentences)
    embedding = pd.DataFrame(sentence_embeddings)
    data2 = pd.concat([data,embedding],axis=1)
    # breakpoint()
    data2.to_csv(output_path,index=False)


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description='Embedding creation for queries')
    argument_parser.add_argument('--path_input', type=str,default="sample_query.txt", help='Path Name', required=False)
    argument_parser.add_argument('--path_output', type=str,default="query_output.csv", help='Path Name', required=False)
    arguments = argument_parser.parse_args()
    input_path = arguments.path_input
    output_path = arguments.path_output
    create_embedding(input_path,output_path)



