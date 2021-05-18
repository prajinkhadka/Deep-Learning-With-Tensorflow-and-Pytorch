import torch 
import torch.nn.functional as F 

## BASIC SELF ATTENSION 

# Suppose we have an input sequence of t vectors with dimension k -> so the size woould be matrix of (t, k)
# Building a minibatch b will give dimension -> (b, t, k)

x = ... 
weights_ = torch.bmm(x, x.transpose(1,2))

weights = F.softmax(weights_, dim=2)

# output  
y = torch.bmm(weights, x)

# size of y is -> (b, t, k)
# The rows are weighted sum over x 





# attention with full features 
import torch 
from torch import nn 
import torch.nn.functional as F 

class SelfAttention(nn.Module):
    def __init__(self, k, heads=8):
        super().__init__()
        self.k = k 
        self.heads = heads 
        # k -> dimension
        self.toKeys = nn.Linear(k, k *heads, bias=False)
        self.toQueries = nn.Lineark, k* heads, bias=False)
        self.toValues = nn.Linear(k, k*heads, bias=False)

        # Unifying output of different heads into single vector 
        self.unifyheads = nn.Linear(heads*k, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads 

        # The output of linear will be -> b, t, h*k  
        queries = self.toQueries(x).view(b, t, h, k)
        keys = self.toKeys(x).view(b, t, h, k)
        values = self.toValues(x).view(b, t, h, k)

        keys = keys.transpose(1, 2).contigous().view(b*h, t, k)
        queries = queries.transpose(1,2).contigous().view(b*h, t, k)
        values = values.transpose(1,2).contigous().view(b*h, t, k)

        queries = queries / (k ** (1/4))
        keys    = keys / (k ** (1/4))

        dot = torch.bmm(queries, keys.transpose(1,2))
        # dot -> size -> (b*h, t, t)
        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, k)

        # to make sure -> head and embedding dim. are next to each other.
        out = out.transpose(1,2)
        out= out.contigous().view(b, t, h*k)
        
        return self.unifyheads(out)



class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        self.attention = SelfAttention(k, heads=heads)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4*k),
            nn.ReLU(),
            nn.Linear(4*k,k)
        )

    def forward(self, x):
        attented = self.attention(x)
        x = self.norm1(attented + x)
        fedforward = self.ff(x)
        return self.norm2(fedforward + x)




class Transformer(nn.Module):
    def __init__(self, k , heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()
        self.num_tokens = num_tokens 
        self.token_emb = nn.Embedding(num_tokens, k )
        self.pos_emp = nn.Embedding(seq_length, k)

        tblocks = []
        for i in range(depth):
            tblocks.apppend(TransformerBlock(k=k, heads=heads))

        self.tblocks = nn.Sequential(*tblocks)
        self.toprobs = nn.Linear(k, num_classes)
        
    def forward(self, x):
        """
        param x: A (b ,t) -> tensor of integer values represneting the words in some predetermiend vocab.

        returns (b, c ) -> tensor of log likelihood prob over clases/
        """
        tokens = self.token_emb(x)
        b, t, x = tokens.size()

        positions= torch.arange(t)
        positions = self.pos_emb(positions)[None,:, :].expand(b, t, k)
        x = tokens + positions
        x = self.tblocks(x)

        # av rg pool over  t  dim.
        x = self.toprobs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)
        