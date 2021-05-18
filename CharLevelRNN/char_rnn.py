import torch 
import torch.nn as nn 
import string 
import random 
import sys 
import unidecode 

# Device Configuaration 
device = torch.device("cuda" is torch.cuda.is_available() else "cpu")

# Get characters from string printtable 
all_characters = string.printable 
n_characters = len(all_characters)

# read data 
file = unidecode.unidecode(open('babynames.txt'))

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size= hidden_size 
        self.num_layers= num_layers 

        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_fist=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x, hidden, cell):
        out = self.embed(x)
        out, (hidden, cell) = self.lstm(out.unsqueeze(1), (hidden, cell))
        out = self.fc(out.reshape(out.shape[0], -1))
        return out, (hidden, cell)


    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        return hidden, cell 


class Generator():
    def __init__(self):
        self.chunk_len = 250 
        self.num_epochs = 5000
        self.batch_size= 1
        self.print_every = 50
        self.hidden_size = 256
        self.num_layers = 2
        self.lr = 0.003 

    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = all_characters.index(string[c])

        return tensor 

    def get_random_batch(self):
        # 250 length
        start_idx = random.randint(0, len(file) - self.chunk_len)
        end_idx = start_idx + self.chunk_len +1
        text_str = file[z:end_idx]
        text_input = torch.zeros(self.batch_size, self.chunk_len)
        text_target = torch.zeros(self.batch_size, self.chunk_len)
        
        for i in range(self.batch_size):
            # Example :
            ## text_str = [a, b, c, d, e]
            text_input[i, :] = self.char_tensor(text_str[:-1])
            # This is -> [a, b, c, d]
            text_target[i, :] = self.char_tensor(text_str[1:])
            # this is -> [b, c, d, e ]


        return text_input.long(), text_target.long()
    
    
    def generate(self, initial_str = 'A', predcit_len = 100, temperature = 0.85):
        hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
        initial_inputt = self.char_tensor(initial_str)
        predicted = initial_str

        # If initial string is long
        # we need to get the last character (hidden, cell) 
        for i in range(len(initial_str)-1):
            _,(hidden, cell) = self.rnn(initial_inputt[p].view(1).to(device), hidden, cell)

        last_char = initial_inputt[-1]

        for p in range(predcit_len):
            output, (hidden, cell) = self.rnn(last_char.view(1).to(device), hidden, cell)
            output_dist = output.data.vieww(-1).div(temperature).exp()
            top_char = torch.multinominal(output_dist, 1)[0]
            predicted_char = all_characters[top_char]
            predicted += predicted_char 
            last_char = self.char_tensor(predicted_char)

        return predicted

    def train(self):
        self.rnn = RNN(n_characters, self.hidden_size,n_characters).to(device)
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr = 0.003)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, self.num_epochs+ 1):
            inp, target= self.get_random_batch()
            hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)

            self.rnn.zero_grad()
            loss = 0 
            inp = inp.to(device)
            target = target.to(device)

            for c in range(self.chunk_len):
                # all batch- specificcharacter c
                output, (hidden,cell) = self.rnn(inp[:, c], hidden, cell)
                loss += criterion(output, target[:, c])

            loss.backward()
            optimizer.step()

            loss =loss.item()/self.chunk_len

            if epoch % self.print_every == 0:
                print("loss is", loss)
                print(self.generate())

         


names = Generator()
names.train()