"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

"""
import math
import copy
import os
import spacy
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V
    """
    d_k = Q.size(-1)
    logits = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        logits = logits.masked_fill(mask, float("-inf"))

    attn_w = F.softmax(logits, dim=-1)
    output = torch.matmul(attn_w, V)
    return output, attn_w

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    # Create a mask where padded token positions are marked True and expand dimensions for attention compatibility
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)

def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:

    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    batch, tgt_len = tgt.shape
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device),
        diagonal=1,
    )

    # combine: True if EITHER pad OR future  -> [batch, 1, tgt_len, tgt_len]
    return pad_mask | causal_mask

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, use_scaling: bool = True) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        self.use_scaling = use_scaling          # Whether to apply scaling in attention score computation
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_q, _ = query.shape
        seq_k = key.size(1)

        # [batch, seq, d_model] -> [batch, seq, num_heads, d_k] -> [batch, num_heads, seq, d_k]
        Q = self.W_Q(query).view(batch, seq_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key  ).view(batch, seq_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch, seq_k, self.num_heads, self.d_k).transpose(1, 2)

        # scaled dot-product attention for all heads in parallel
        logits = torch.matmul(Q, K.transpose(-2, -1))
        if self.use_scaling:
            logits = logits / (self.d_k ** 0.5)
        if mask is not None:
            logits = logits.masked_fill(mask, float("-inf"))
        attn_w = F.softmax(logits, dim=-1)
        self.attn_weights = attn_w.detach()
        attn_w = self.dropout(attn_w)
        heads  = torch.matmul(attn_w, V)   

        # [batch, num_heads, seq_q, d_k] -> [batch, seq_q, num_heads, d_k] -> [batch, seq_q, d_model]
        out = heads.transpose(1, 2).contiguous().view(batch, seq_q, self.d_model)
        return self.W_O(out)

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        seq_len   = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)   # [1, seq_len]
        return self.dropout(x + self.pos_emb(positions))                  # broadcasts over batch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)   # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )                                                                      # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)   # even dims
        pe[:, 1::2] = torch.cos(position * div_term)   # odd  dims

        # add a batch dim so it broadcasts: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # register_buffer = state but NOT a learnable parameter; moves with .to(device)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.p = dropout
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #2 linear layers with ReLU activation
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x
    
class EncoderLayer(nn.Module):
    #x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        attn = self.self_attn(x,x,x, mask=src_mask)
        x = self.norm1(x + self.dropout(attn))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]
        """
        masked_attn = self.self_attn(x,x,x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(masked_attn))
        cross_attention = self.cross_attn(x,memory,memory, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_attention))
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        return x

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        # Create N independent copies of the encoder layer so each layer has its own learnable parameters
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        # Create N independent copies of the decoder layer so each layer has its own learnable parameters
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.
    """

    def __init__(
        self,
        src_vocab_size: int = 7853,
        tgt_vocab_size: int = 5893,
        d_model:   int   = 256,         # ← was 512
        N:         int   = 3,           # ← was 6
        num_heads: int   = 8,
        d_ff:      int   = 1024,        # ← was 2048
        dropout:   float = 0.1,
        use_scaling: bool = True,  
         learned_pos: bool = False,  
        checkpoint_path: str = "checkpoint.pt",
        gdrive_id: str = "1lBO4zzf1Xc-9XJvCVB6hq0W-JzVnpPC9",
    ) -> None:
        super().__init__()

        # token embeddings (scaled by sqrt(d_model) in forward, per paper §3.4)
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=1)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=1)
        self.d_model   = d_model

        # positional encoding (shared module — it's stateless other than the buffer)
        self.pos_enc = (
            LearnedPositionalEmbedding(d_model, dropout=dropout)
            if learned_pos
            else PositionalEncoding(d_model, dropout=dropout)
        )

        # encoder / decoder stacks built from template layers
        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)

        if not use_scaling:
            for layer in self.encoder.layers:
                layer.self_attn.use_scaling = False
            for layer in self.decoder.layers:
                layer.self_attn.use_scaling  = False
                layer.cross_attn.use_scaling = False

        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # Xavier init for all >1-D parameters (paper-standard init)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # tokenizers — loaded lazily in .infer() so Transformer() works
        # on bare containers (e.g. autograder) without spacy models installed
        self.spacy_de = None
        self.spacy_en = None

        # vocabs — filled in from checkpoint
        self.src_vocab = None
        self.src_itos  = None
        self.tgt_vocab = None
        self.tgt_itos  = None
    
        if not os.path.exists(checkpoint_path):  #to download from gdown
            try:
                gdown.download(id=gdrive_id, output=checkpoint_path, quiet=False)
            except Exception as e:
                print(f"[Transformer] could not download checkpoint: {e}")

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(ckpt["model_state_dict"])
            self.src_vocab = ckpt.get("src_vocab")
            self.src_itos  = ckpt.get("src_itos")
            self.tgt_vocab = ckpt.get("tgt_vocab")
            self.tgt_itos  = ckpt.get("tgt_itos") 

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.src_embed(src) * math.sqrt(self.d_model)   # [B, src_len, d_model]
        x = self.pos_enc(x)                                  # add PE + dropout
        return self.encoder(x, src_mask)  

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)  # [B, src_len, d_model]
        x = self.pos_enc(x)                                # add PE + dropout
        x = self.decoder(x,memory,src_mask,tgt_mask)
        return self.generator(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)   #take contetx vector as encoder output
        return self.decode(memory, src_mask, tgt, tgt_mask)


    def _tokenize_de(self, sentence: str) -> list[str]:
        """
        Tokenize a German sentence the same way training did.
        Tries spacy first; falls back to a regex tokenizer that approximates
        spacy's behaviour (whitespace + punctuation split) when the spacy
        model isn't installed (e.g. on a barebones autograder container).
        """
        if self.spacy_de is None:
            try:
                self.spacy_de = spacy.load("de_core_news_sm")
            except OSError:
                self.spacy_de = False  # sentinel: spacy unavailable, use regex
        if self.spacy_de:
            return [t.text.lower() for t in self.spacy_de.tokenizer(sentence)]
        import re
        return re.findall(r"\w+|[^\w\s]", sentence.lower(), flags=re.UNICODE)

    def infer(self, src_sentence: str) -> str:
        # Import special token indices used in the dataset
        from dataset import SOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX

        if self.src_vocab is None or self.tgt_itos is None:
            raise RuntimeError("Vocabs not loaded — checkpoint missing or corrupt.")

        self.eval()

        device = next(self.parameters()).device
        toks = self._tokenize_de(src_sentence)
        # Convert tokens to vocabulary indices
        # Add SOS (start of sentence) and EOS (end of sentence) tokens
        src_ids = [SOS_IDX] + [self.src_vocab.get(t, UNK_IDX) for t in toks] + [EOS_IDX]

        # Convert list of token IDs into tensor and add batch dimension
        src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = make_src_mask(src, pad_idx=PAD_IDX)

        # Encode the source sentence into memory representations
        with torch.no_grad():
            memory = self.encode(src, src_mask)

        # Initialize target sequence with SOS token
        ys = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)

        # Greedy decoding loop (maximum 99 tokens)
        for _ in range(99):
            tgt_mask = make_tgt_mask(ys, pad_idx=PAD_IDX)
            with torch.no_grad():
                logits = self.decode(memory, src_mask, ys, tgt_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == EOS_IDX:
                break

        out_ids = ys.squeeze(0).tolist()[1:]
        if EOS_IDX in out_ids:
            out_ids = out_ids[: out_ids.index(EOS_IDX)]

        # Convert token IDs back to words
        # Ignore padding and SOS tokens
        tokens = [self.tgt_itos[i] for i in out_ids if i not in (PAD_IDX, SOS_IDX)]

        return " ".join(tokens)