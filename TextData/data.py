import spacy 
from torchtext.datasets import Multi30k 
from torchtext.data import Field, BucketIterator

spacy_en = spacy.load('en')

spacy_ger = spacy.load('ger')

def tokenize_eng():
    return [tok.text for tok in spacy_eng.tokenizer(text)]


def tokenize_ger():
    return [tok.text for tok in spacy_ger.tokenizer(text)]



english = Field(sequential=True, use_vocab=True, tokenize_eng=tokenize_eng, lower=True)
german = Field(sequential=True, use_vocab=True, tokenize_eng=tokenize_ger, lower=True)

train_data, validation_data, test_data = Multi30k.split(exts=('.de', '.en'), 
                                                        fields = (german, english))

english.build_vocab(train_data, max_size=10000, min_freq =2)

german.build_vocab(train_data, max_size=10000, min_freq =2)


train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size = 64, 
    device ='cuda'
)

for batch in train_iterator:
    print(batch)