import spacy 
import pandas as pd 
from torchtext.data import Field, BucketIterator, TabularDataset 
from sklearn.model_selection import train_test_split 

english_txt = open('train_WMT_english.txt', encoding ='utf-8').read().split("/n")
german_txt = open('train_WMT_german.txt', encoding ='utf-8').read().split("/n")

raw_data = {'English': [line for line in english_txt[1:100]],
            'German': [line for line in german_txt[1:100]]}


df = pd.DataFrame(raw_data, columns=['English', 'German'])

train, test = train_test_split(df, test_size=0.2)

train.to_json('train.json', orient='records', lines=True)
test.to_json('test.json', orient='records', lines=True)

train.to_csv('train.csv', index=False)
test.to_csv('test.csv',index=False)


