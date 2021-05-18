import torch 
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim 

class CatDogDataset(Dataset):
    def __init__(csvfile, root_dir,transform ):
        self.annotations = pd.read_csv(csvfile)
        self.root_dir = root_dir 
        self.transform = transform 

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self,index):
        img_path = os.path.join(self.root_dir,self.annotations.iloc[index])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)
        
        return (image, y_label)





dataset = CatDogDataset(csvfile= "file.csv", root_dir = "rootdirectort", transform = transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [2000, 500])
train_loader = DataLoader(dataset=train_set, batch_size =64, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=False )