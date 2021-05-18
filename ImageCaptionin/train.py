import torch 
import torch.nn as nn 
import torch.optim as optim 
import torchvision.transforms import transforms 

from loader import get_loader 
from model import CNNtoRNN 

def train():
    transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_loader = get_loader(
        root_folder= "flickt8k/images",
        annotation_file = "flickt8k/captions.txt",
        transform= transform
    )

    torch.backends.cudnn.benchmark = True 
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    # hyper 
    embed_size = 256  
    hidden_size= 256 
    vocab_size = len(dataset.vocab)
    num_layers = 1 
    learning_rate = 3e-4 
    num_epochs=100 

    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.voca.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train() 
    for epoch in range(num_epochs):
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(), 
                "optimzier": optimizer.state_dict(),
                "step": step
            }

        for idx, (imgs, captions ) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1]) # not sending end tpoken
            loss = criterion(outputs.reshape(-1, output.shape[2]), captions.reshape(-1, output.shape[2]))
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()


if __name__ == '__main__':
    train()

