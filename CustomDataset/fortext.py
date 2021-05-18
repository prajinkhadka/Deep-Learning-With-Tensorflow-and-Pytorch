# Task 
# 1. Convert text to numerical values. This
# 2. need vocabulary mapping , each ord to index 
# 3. Setup pytorch Datset to laod the data and
# 4. setup padding ( all the xample should be of same length)

import os 
import spacy 
import torch 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset 
from PIl import Image 
import torchvision.transforms as transforms
class Vocabulary:
    def __init__(self, freq_thresholds):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2 : "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "UNK": 3}

        self.freq_thresholds = freq_thresholds

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() in for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4 
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] =1 
                else:
                    frequencies[word] += 1 
                
                if frequencies[word] == self.freq_thresholds:
                    self.stoi[word] = idx 
                    self.itos[idx] = word 
                    idx +=1 

    def numerizalize(self, text):
        tokenized_text = self.tokenizer_eng() 
        return [
            self.stoi[token] if tokein in self.stoi else self.stoi['<UNK>'] for token in tokenized_text
        ]



class Flickr(Dataset):
    def __init__(self, root_dir, caption_file, transform=None, freq_thresholds):
        self.root_dir = root_dir
        self.df = pd.read_csv("captionsfile")
        self.transform = transform 

        # Get image, caption colums 
        self.imgs = self.df['image']
        self.captions = self.df['captions']

        self.vocab = Vocabulary(freq_thresholds)
        self.vocab.build_vocabulary(self.captions.tolist())

    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os,path.join(self.root_dir, img_id)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        numerical_caption = [self.vocab.stoi["<SOS>"]]
        numerical_caption += self.vocab.numerizalize(caption)
        numerical_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numerical_caption)
 


 class MyCollate():
     def __init__(self, pad_idx):
         self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeuze(0) for item in batch]
        imgs = torch.cat(imgs, dim =0)

        targets =[ item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        return imgs, targets 



def get_loader(
    root, 
    annotations,
    transform,
    batch_size=32,
    num_workers=4, 
    shuffle=True,
    pin_memory =True):

    dataset = Flickr(root_folder, annotations, transform= transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(dataset=dataset, batch_size= batch_size, num_workers=num_workers, shuffle= shuffle,
                            shuffle=shuffle, pin_memory=pin_memory,
                            collate_fn = MyCollate(pad_idx=pad_idx)
                            )

    return loader 



transform = transforms.Compose (
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]
)
dataloader = get_loader("flickr/8k, ""flickr/captions.txt", transform=transform)

for idx, (imgs, captions) in enumerate(dataloader):
    print(imgs.shape)
    print(captions.shape)



    

