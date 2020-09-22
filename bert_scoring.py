
import argparse
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, BertEmbeddings
from flair.data import Sentence
import numpy as np
import os
from configparser import ConfigParser
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

def get_config(config_section, config_name, config_path, is_boolean=False, is_int=False, is_float=False):
    """
    :param config_section: part of config file to be parsed
    :param config_name: name of config parameter to read
    :param config_path: full config file path including file name. Relative file path should also work
    :param is_boolean : Bool : whether the value is boolean or not
    :param is_int : Bool : whether the value is integer or not
    :param is_float : Bool : whether the value is float or not
    :return: config value
    """
    parser = ConfigParser()
    parser.read(config_path)
    if is_boolean:
        return parser.getboolean(config_section, config_name)
    elif is_int:
        return parser.getint(config_section, config_name)
    elif is_float:
        return parser.getfloat(config_section, config_name)
    else:
        return parser.get(config_section, config_name)

# myembed={}
# def create_embeddings(path_name="sample_query.txt"):
#     myembed={}
#     with open("sample_query.txt","r") as f:
#         for l in f:
#             query=l.replace("\n","").strip()
#             query2 = Sentence(query)
#             document_embeddings.embed(query2)
#             embed=np.array(query2.embedding)
#             myembed[query]=list(embed)
#
#     embed_len=embed.shape[0]
#     embed_pandas=pd.DataFrame.from_dict(myembed, orient='index',columns=["dim_"+str(i) for i in range(embed_len)])
#
#     embed_pandas.to_csv("write.csv",index=False)


def compute_pretrained_individual_transformer_embedding(query, word_embedding, document_embeddings):
    """
       :param query: String :: arbitrary sentence
       :param word_embedding
       :param document_embeddings
       :return: n-dimensional embedding
    """
    tokenized_text = word_embedding.tokenizer.tokenize(query)
    tokenized_len = len(tokenized_text)
    while tokenized_len > 512:
        query = query[:len(query)//2]
        tokenized_text = word_embedding.tokenizer.tokenize(query)
        tokenized_len = len(tokenized_text)
    sentence = Sentence(query)
    document_embeddings.embed(sentence)
    tensor_query_embedding = sentence.get_embedding()
    numpy_query_embedding = tensor_query_embedding.data.cpu().numpy()
    return numpy_query_embedding


##################################################################################
def compute_query_embedding_pretrained_models(query):

    """
        learn bert sentence embeddings using pre-trained word embedding model for an arbitrary question
       :param query: String :: arbitrary sentence
       :return: n-dimensional embedding
    """
    query = query
    up = os.path.abspath(os.path.dirname(__file__))
    conf_path = os.path.join(up, 'config.ini')
    embedding_name = 'bert'
    transformer_model_or_path = get_config('Transformers', '{}_model_or_path'.format(embedding_name), conf_path, 0)
    transformer_layers = get_config('Transformers', '{}_layers'.format(embedding_name), conf_path, 0)
    transformer_pooling_operation = get_config('Transformers', '{}_pooling_operation'.format(embedding_name), conf_path,0)
    transformer_use_scalar_mix = get_config('Transformers', '{}_use_scalar_mix'.format(embedding_name), conf_path, 1)

    word_embedding = BertEmbeddings(bert_model_or_path=transformer_model_or_path,
                                    layers=transformer_layers,
                                    pooling_operation=transformer_pooling_operation,
                                    use_scalar_mix=transformer_use_scalar_mix
                                    )
    document_embeddings = DocumentPoolEmbeddings([word_embedding], fine_tune_mode='none')
    query_embedding = compute_pretrained_individual_transformer_embedding(query, word_embedding, document_embeddings)
    return query + ',' + str(list(query_embedding))[1:-1]

# def create_embeddings(path_name):
#     myembed = {}
#     count = 0
#     with open(path_name, "r") as f:
#         queries=f.readlines()
#         for query in queries:
#             count +=1
#             logging.debug('Starting to Create embedding for : ' + query + '<---> Record Number : ' + str(count))
#             embed = compute_query_embedding_pretrained_models(query)
#             myembed[query] = list(embed)
#             logging.debug('Finished Creating embedding for : ' + query + '<---> Record Number : ' + str(count))
#     embed_len = embed.shape[0]
#     embed_pandas = pd.DataFrame.from_dict(myembed, orient='index',
#                                           columns=["dim_" + str(i) for i in range(embed_len)])
#
#     up = os.path.abspath(os.path.dirname(__file__))
#     embed_pandas.to_csv(up + "write.csv", index=True)

import time
def time_it(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) )
        else:
            print('%r  %2.2f s' % (method.__name__, (te - ts) ))
        return result
    return timed

@time_it
def create_embeddings_threaded(path_name):
    count = 0
    with open(path_name, "r") as f, open(path_name.split('.')[0] + '_write.csv' , "w") as fw:
        queries=f.readlines()
        num_query=len(queries)
        pool = ThreadPoolExecutor(num_query)
        myembed = []
        for v in queries:
            count +=1
            logging.debug('Starting to Create embedding for : ' + v + ' <---> Record Number : ' + str(count))
            print('Starting to Create embedding for : ' + v + ' <---> Record Number : ' + str(count) + ' completed ' + str(count/num_query*100) + '%')
            myembed.append(pool.submit(compute_query_embedding_pretrained_models, v))
        for f in as_completed(myembed):
            fw.write(f.result() + '\n')
@time_it
def create_embeddings_multiprocess(path_name):
    count = 0
    with open(path_name, "r") as f, open(path_name.split('.')[0] + '_write.csv' , "w") as fw:
        queries=f.readlines()
        num_query=len(queries)
        pool = ProcessPoolExecutor(4)
        myembed = []
        for v in queries:
            count +=1
            v = v.replace('\n','').strip()
            logging.debug('Starting to Create embedding for : ' + v + ' <---> Record Number : ' + str(count))
            print('Starting to Create embedding for : ' + v + ' <---> Record Number : ' + str(count))
            myembed.append(pool.submit(compute_query_embedding_pretrained_models, v))
            for f in as_completed(myembed):
                fw.write(f.result() + '\n')

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description='Embedding creation for queries')
    argument_parser.add_argument('--path_name', type=str,default="sample_query.txt", help='Path Name', required=False)
    arguments = argument_parser.parse_args()
    path_name = arguments.path_name
    # create_embeddings(path_name=path_name)
    create_embeddings_threaded(path_name=path_name)
    # create_embeddings_multiprocess(path_name=path_name)
