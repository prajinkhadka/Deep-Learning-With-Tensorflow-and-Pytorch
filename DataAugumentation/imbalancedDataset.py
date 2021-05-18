import torch 
import torchvision.datasets as datasets
import os 
from torch.utils.data import WeightedRandomSampler, DataLoader #
import torchvision.transforms as transforms #
import torch.nn as nn #


# Oversampling - 
# Oversampel t he single image by data augmentations and othe techniques.

# Class weighting. 
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1, 50]))

# OVersampling -

def get_loader(root_dir, batch_size):
    my_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=root_dir, transform=my_transform)

    class_weights = []

    for root,subdir, files in os.walk(root_dir):
        if len(files) > 0:
            class_weights.append(1/len(files))

    # class_weights = [1, 50]  

    # In WeightedRandomSampler, we need to specify weights for each of the file in dataset.
    # So, how we do that is, first create a sample weight with 0 for all files in dataset.
    sample_weights = [0] * len(dataset)

    for idx, (data,label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)
    return loader 




def main():
    pass 

if __name__ == "__main__":
    main() 