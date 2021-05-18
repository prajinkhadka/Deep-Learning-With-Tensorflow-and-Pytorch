import torch 
import torch.nn as nn 
import torchvision
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN 
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, images):
        features= self.inception(images)

        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True 
            else:
                param.requires_grad = self.train_CNN 
        
        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers) 
        self.linear= nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _  = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs 

    
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images) 
        outputs = self.decoderRNN(features, captions)
        return outputs  

    # In case of Image Captioning, there is a distinction between training and testing. 
    # In case of training, while sedning inputs to lstm to send the the actual data from dataset -> Teacher Forcing.
    # In cse of testing, we send the predicted word. 


    def caption_image(self, image, vocabulary, max_lenghth=50):
        result_caption =[]
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None 
            for _ in range(max_lenghth):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeuze(0))

                predicted = output.argmax(x)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeuze(0)

                if vocabulary.itos[predicted.item()== "<EOS>"]:
                    break 
                    
        return [vocabulary.itosp[idx] for idx in result_caption]


