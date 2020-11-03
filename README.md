# Latest using sentence transformers

 pip install sentence_transformers
 
 python bert_scoring4.py --path_input <input_path> --path_output <output_path>

Default Input Path - "sample_query.txt" 

Default output Path - "query_output.csv"



# bert_scoring
How to vectorize sentences using bert

https://bert-as-service.readthedocs.io/en/latest/section/get-start.html



If above does not suit you. Following will help without creating server instance.

For scoring file 1 and 2:


Note that we have used bert-based uncased. You can change in the config file

Steps to run:
1. Download these two files and keep in a directory . mkdir ~/test
2. Open cmd
3. Go to the dir. cd ~/test
4. run : "python bert_scoring.py --path_name "your_file.txt" 

path name is optional , default file is "sample_query.txt"     

output file is "write.csv" in the working directory


For file 3:

pip install transformers

if installation does not work

git clone https://github.com/huggingface/transformers.git

cd transformers

pip install -e .

python bert_scoring3.py --path_input sample.csv --path_output output.csv


Bert Models:
ID	Language	Embedding
'bert-base-uncased'	English	12-layer, 768-hidden, 12-heads, 110M parameters
'bert-large-uncased'	English	24-layer, 1024-hidden, 16-heads, 340M parameters
'bert-base-cased'	English	12-layer, 768-hidden, 12-heads , 110M parameters
'bert-large-cased'	English	24-layer, 1024-hidden, 16-heads, 340M parameters
'bert-base-multilingual-cased'	104 languages	12-layer, 768-hidden, 12-heads, 110M parameters
'bert-base-chinese'	Chinese Simplified and Traditional	12-layer, 768-hidden, 12-heads, 110M parameters

