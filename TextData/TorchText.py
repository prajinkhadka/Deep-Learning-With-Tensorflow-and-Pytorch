# What Torch Text can do  ?
# 1. Train / Validation / Test 
# 2. File loading from different sources.
# 3. Toeknization - Break list of sentences into words.
# 4. Vocab - Generate Vocabulary List 
# 5. Numericalzie/ Identify : Map Words into integer numbers for the entire corpus.
# 6. Word Vector - Either initialzie voacablary randomly oe load some pretrained embedding, 
    # the embedding msut be trimmed, meaning we only store words in our vocabualary into memory.

# Batching - Generate batch 
# Embedding Lookup - Max each sentences ith contain word indices to fixed dimenstion ord vector.s

### What we want to do ####    

# Sentence = "The qucik brown fox jumped over a lazy dog."

# Toeknization ->> ["The", "qucik", "brown" ,"fox", "jumped",  "over", "a", "lazy" "dog"] 
# Vocabulary -->> {"The" -> 0, 
#                 "qucik" ->1, 
#                 "brown" -> 2, 
#                 "fox" -> 3,
#                  "jumped" -> 4, 
#                  "over" -> 5,
#                   "a" -> 6, 
#                   "lazy" -> 7,
#                    "dog" -> 8}

# Numerizalization = [0, 1, 2, ... 8]

# embedding Lookcup : [ 
#                   [0.3, 0.5, 0.4],
#                    [0.8, 0.2, 0.4],
#                    ...
#                    ]

# Steps 
# 1. Secify ho preprocessing is done - > Fields.
# 2. Use dataset to laod the data.  -> Tabualr Datst. 
# 3. Contruct an inetor to batching and padding - > BucketIterator
import torchtext 
from torchtext.data.legacy.data import Field, TabularDataset, BucketIterator 
import spacy 

tokenize = lambda x: x.split() 
quote = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True) 
score = Field(sequential=False, use_vocab=False)

fields = {'quote':('q', quote), 'score':('s', score)}

train_data, test_data= TabularDataset.splits(
                                    path ='mydata',
                                    train ='train.json',
                                    test=  'test.json',
                                    format ='json',
                                    fields = fields
                                )


spacy_en = spacy.load('en')
def tokenize(text):
    return [tok.text for tok in space_en.tokenizer(text)]


quote.build_vocab(train_data, 
                    max_size=10000,
                    min_freq=1)


train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=2,
    device="cpu"
)

for batch in train_iterator:
    print(batch.q)