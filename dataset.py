"""
dataset.py — Multi30k loading, spaCy tokenization, vocab building.
DA6401 Assignment 3: "Attention Is All You Need"

Special tokens (fixed order — pad_idx=1 matches make_src_mask default in model.py):
    <unk> = 0   <pad> = 1   <sos> = 2   <eos> = 3
"""

from collections import Counter
from datasets import load_dataset
import spacy


UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]
MIN_FREQ = 2


class Multi30kDataset:
    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        self.data = load_dataset("bentrevett/multi30k", split=split)

        # spaCy tokenizers for German (src) and English (tgt)
        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")

        # vocabs are built lazily by build_vocab()
        self.src_vocab = None   # dict: token -> index   (stoi)
        self.tgt_vocab = None
        self.src_itos  = None   # list: index -> token   (itos)
        self.tgt_itos  = None

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        src_counter, tgt_counter = Counter(), Counter()
        for row in self.data:
            src_counter.update(tok.text.lower() for tok in self.spacy_de.tokenizer(row["de"]))
            tgt_counter.update(tok.text.lower() for tok in self.spacy_en.tokenizer(row["en"]))

        # source (German) vocab
        self.src_itos = list(SPECIALS)
        for tok, c in sorted(src_counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if c >= MIN_FREQ and tok not in SPECIALS:
                self.src_itos.append(tok)
        self.src_vocab = {tok: i for i, tok in enumerate(self.src_itos)}

        # target (English) vocab
        self.tgt_itos = list(SPECIALS)
        for tok, c in sorted(tgt_counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if c >= MIN_FREQ and tok not in SPECIALS:
                self.tgt_itos.append(tok)
        self.tgt_vocab = {tok: i for i, tok in enumerate(self.tgt_itos)}

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary.
        """
        if self.src_vocab is None or self.tgt_vocab is None:
            raise RuntimeError("Call build_vocab() before process_data().")

        processed = []
        for row in self.data:
            src_toks = [tok.text.lower() for tok in self.spacy_de.tokenizer(row["de"])]
            tgt_toks = [tok.text.lower() for tok in self.spacy_en.tokenizer(row["en"])]

            # encode src: <sos> + token_ids + <eos>, unknown words -> <unk>
            src_ids = [SOS_IDX]
            for t in src_toks:
                src_ids.append(self.src_vocab.get(t, UNK_IDX))
            src_ids.append(EOS_IDX)

            # encode tgt: <sos> + token_ids + <eos>
            tgt_ids = [SOS_IDX]
            for t in tgt_toks:
                tgt_ids.append(self.tgt_vocab.get(t, UNK_IDX))
            tgt_ids.append(EOS_IDX)

            processed.append((src_ids, tgt_ids))
        return processed

