from Seq2Seq.seq2seq import Decoder
import torch 
import torch.nn as nn 

class SelfAttention(nn.Moddule):
    # We have embedding size and we need to split differrent parts.
    # In how many parts we split = 8 

    # Suppose we have 256-> embed size 
    # and heads =8 
    # then we split in 8*32 parts.
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads 
        self.head_dim = embed_size // heads 
        assert (self.head_dim * heads == embed_size), "Embedding size should be divislbe hy head"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get the number of training examples 
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Splitting embedding into self.head dim.
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        # Splitting in << Self.heads, Self.head_dim >> 
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values
        keys = self.keys 
        queries = self.queries
        
        # Multiply queries and keyrs
        # n- number of examples, q-> qury len, h->  heads, d-> head dim 
        # querie.shape -> (N, query_len, heads, heads_dim)
        
        energy = torch.einsum("nqhd,nkhd→nhqk", [queries, keys])

        # keys.shape -> (N, key_len, heads, heads_dim)
        # energy -> (N, heads, query_len, key_len)
        
        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))

        attention = torch.softmax(energy/ (self.embed_size ** (1/2)), dim=3)
        # dim=3 means , normalizing in key_len dimenstion 
        # Suppose key_len is source sentece length and query_len is according to target sentence length.
        # Making attenstion socre normalize to 1 ( key_len)
        # if first is 0.6, then it means give 60% attension t first word on setnece.

        out = torch.einsum("nhql,nlhd→nqhd", [attention, values]).reshape(
            N, query_len, self.heads, self.head_dim
        )

        # attentsion shape -> (N, heads, query_len, key_len)
        # values shape -> (N, value_len, head, head_dim)

        # the key_len and value_len is always the same.
        # so we are multiplying in that dimenstion.

        # After Einsum - >out -> (N, query_len, heads, head_dim)
        # and flatten 

        out = self.fc_out(out)
        return out 



class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_extension):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNrom(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_extension*embed_size),
            nn.ReLU(),
            nn.Linear(forward_extension*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,  value, key, query, mask):
        attention= self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention+ query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out 


class Encoder(nn.Module):
    def __init__(
                self,
                src_vocab_size,
                embed_size,
                num_layers,
                heads, 
                device,
                forward_extension,
                dropout,
                max_length 
        ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size 
        self.device = device 
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_emb = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
                [
                    TransformerBlock(
                        embed_size,
                        heads,
                        dropout=dropout,
                        forward_extension= forward_extension
                    )
                for _ in range(num_layers)]
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_len = x.shape
        # expand -> to make it for every example
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_emb(positions))

        for layer in self.layers:
            # value , key and query all gonna be same in encoder.
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_extension, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_extension)
        self.dropout = nn.Dropout(dropout)


    # src_mask -> padding the sohert sentences so that it matches the length of long setnece.
    # x -> target 
    # values, key -> from Encoder
    def forward(self, x, value, key, source_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.tranformer_block(value, key, query, source_mask)

        return out 


class Decode(nn.Module):
    def __init__(self,target_vocab_size,embed_size,num_layers,heads, forward_extension,dropout, device, max_length):
        super(Decode, self).__init__()
        self.device = device 
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_extension, dropout, device)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.drouout = nn.Drouput(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.droput(self.word_embedding(x)+ self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out



class Transformer(nn.Module):
    def __init__(self, src_vocab_size, 
                        trg_vocab_size, 
                        src_pad_idx, 
                        trg_pad_idx, 
                        embed_size=256,
                        num_layers=6,
                        forward_extension=4, 
                        heads=8,
                        dropout=0,
                        device="cuda",
                        max_length=100
    ):
                        super(Transformer, self).__init__()
                        self.encoder = Encoder(
                                                src_vocab_size, 
                                                embed_size, 
                                                num_layers, 
                                                heads, 
                                                device, 
                                                forward_extension, 
                                                dropout, 
                                                max_length)

                        self.decoder = Decoder(
                                                trg_vocab_size,
                                                embed_size,
                                                num_layers, 
                                                heads, 
                                                forward_extension, 
                                                dropout, 
                                                device,
                                                max_length)
                        self.src_pad_idx = src_pad_idx
                        self.trg_pad_idx = trg_pad_idx
                        self.device = device 
    

    def mark_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def mark_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N,1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.mark_src_mask(src)
        trg_mask = self.mark_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)