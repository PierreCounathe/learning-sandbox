import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_length: int, dropout_p: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout_p)
        
        # Create matrix of the desired shape
        batch_positional_encoding = torch.zeros(seq_length, d_model)  # (seq, d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)  # (seq, 1)
        dimension = torch.arange(0, d_model, 2, dtype=torch.float).unsqueeze(0)  # (1, d_model/2)
        batch_positional_encoding[:, ::2] = torch.sin(position / (10000 ** (dimension / d_model)))  # (seq, d_model)
        batch_positional_encoding[:, 1::2] = torch.cos(position / (10000 ** (dimension / d_model)))  # (seq, d_model)
        
        # Add a batch dimension
        batch_positional_encoding = batch_positional_encoding.unsqueeze(0)  # (1, seq, d_model)
        
        self.register_buffer('pe', batch_positional_encoding)
        
    def forward(self, x):
        # :x.shape[1] because we want to add positional encoding where we have input
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
        


class LayerNormalization(nn.Module):
    def __init__(self, epsilon: float = 10e-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        # Dimension of x should be (B, seq, d_model)
        mean = x.mean(axis=-1, keepdim=True)
        std = x.std(axis=-1, keepdim=True)
        return self.alpha * ((x - mean) / (std + self.epsilon)) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_p: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_p)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # x: (B, seq_len, d_model)
        x = torch.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout_p: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "model dimension is not divisible by the number of heads"
        
        self.d_k = self.d_model // self.h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_p)
    
    @staticmethod
    def self_attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # (B, h, seq, seq)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (B, h, seq, seq), dim=1 because we then multiply the last dim of attn by value
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return attention_scores @ value, attention_scores
    
    def forward(self, q, k, v, mask):
        # Compute the query, keys and values from the input and weights
        query = self.w_q(q)  # q (B, seq, d_model) -> query (B, seq, d_model)
        key = self.w_k(k)  # same
        value = self.w_v(v)  # same
        
        # Split query, key, value, into multiple attention heads. Each attention head should have: the whole batch, and full sequences, but a part of the embedding
        # (B, seq, d_model) -> (B, seq, h, d_k) -> (B, h, seq, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.self_attention(query, key, value, mask, self.dropout)  # (B, h, seq, d_k)
        
        # (B, h, seq, d_k) -> (B, seq, h, d_k) -> (B, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)  # (B, seq, d_model)
        
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout_p: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout_p: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout_p) for _ in range(2)])
    
    def forward(self, x, source_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, source_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout_p: float) -> None:
        super().__init__()
        self.self_atention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout_p) for _ in range(3)])
    
    def forward(self, x, encoder_output, source_mask, target_mask):
        x = self.residual_connections[0](x, lambda x: self.self_atention_block(x, x, x, target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, source_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (B, seq, d_model) -> (B, seq, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)
    

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, source_embedding: InputEmbedding, target_embedding: InputEmbedding, source_position: PositionalEncoding, target_position: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.source_position = source_position
        self.target_position = target_position
        self.projection_layer = projection_layer
    
    def encode(self, source, source_mask):
        source = self.source_embedding(source)
        source = self.source_position(source)
        return self.encoder(source, source_mask)

    def decode(self, encoder_output, source_mask, target, target_mask):
        target = self.target_embedding(target)
        target = self.target_position(target)
        return self.decoder(target, encoder_output, source_mask, target_mask)
    
    def project(self, x):
        return self.projection_layer(x)

def build_transformer(
    source_vocabulary_size: int,
    target_vocabulary_size: int,
    source_seq_len: int,
    target_seq_len: int,
    d_model: int = 512,
    N: int = 6,  # TODO: revert back to 6
    h: int = 8,
    dropout_p: float = 0.1,
    d_ff: int = 2048
):
    source_embedding = InputEmbedding(d_model, source_vocabulary_size)
    target_embedding = InputEmbedding(d_model, target_vocabulary_size)
    
    source_position = PositionalEncoding(d_model, source_seq_len, dropout_p)
    target_position = PositionalEncoding(d_model, target_seq_len, dropout_p)
    
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout_p)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout_p)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout_p)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout_p)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout_p)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout_p)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout_p)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, target_vocabulary_size)
    
    transformer = Transformer(encoder, decoder, source_embedding, target_embedding, source_position, target_position, projection_layer)
    
    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        
    return transformer
