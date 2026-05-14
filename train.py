"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
wandb.login(key="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")


from model import Transformer, make_src_mask, make_tgt_mask
from dataset import Multi30kDataset, PAD_IDX, SOS_IDX, EOS_IDX
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class _PairsDataset(Dataset):
    """Wraps Multi30kDataset.process_data() output for use with DataLoader."""
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.pairs[idx]
        return (torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long))


def _collate(batch):
    """Pad src/tgt to longest in batch with PAD_IDX."""
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded

from lr_scheduler import NoamScheduler

from sacrebleu import corpus_bleu


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.eps = smoothing
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1) build smoothed target distribution of shape [N, vocab_size]
        true_dist = torch.full_like(logits, self.eps / (self.vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.eps)
        true_dist[:, self.pad_idx] = 0.0

        # 2) zero out rows where the gold token is <pad>
        pad_mask = (target == self.pad_idx)
        true_dist.masked_fill_(pad_mask.unsqueeze(1), 0.0)

        # 3) KL( true_dist || softmax(logits) )  — batchmean reduction
        log_probs = F.log_softmax(logits, dim=-1)
        return F.kl_div(log_probs, true_dist, reduction="batchmean")

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    model.train() if is_train else model.eval()

    total_loss, total_tokens = 0.0, 0

    for src, tgt in data_iter:
        src = src.to(device)                       # [B, src_len]
        tgt = tgt.to(device)                       # [B, tgt_len]

        # teacher forcing: feed tgt[:-1], predict tgt[1:]
        tgt_in  = tgt[:, :-1]                      # decoder input  (drop last token)
        tgt_out = tgt[:, 1:]                       # gold targets   (drop <sos>)

        src_mask = make_src_mask(src, pad_idx=1)
        tgt_mask = make_tgt_mask(tgt_in, pad_idx=1)

        # forward
        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_in, src_mask, tgt_mask)         # [B, tgt_len-1, V]
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),                # [B*(tgt_len-1), V]
                tgt_out.reshape(-1),                                # [B*(tgt_len-1)]
            )

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        # accumulate weighted by non-pad token count for an honest average
        n_tokens     = (tgt_out != 1).sum().item()
        total_loss  += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    mode = "train" if is_train else "eval"
    print(f"[epoch {epoch_num}] {mode} loss = {avg_loss:.4f}")
    return avg_loss

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()

    # encoder runs once; its output is reused at every decode step
    with torch.no_grad():
        memory = model.encode(src, src_mask)

    # start with just <sos>; shape [1, 1]
    ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx=1)
        with torch.no_grad():
            logits = model.decode(memory, src_mask, ys, tgt_mask)   # [1, cur_len, V]
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)    # [1, 1]
        ys = torch.cat([ys, next_tok], dim=1)
        if next_tok.item() == end_symbol:
            break

    return ys


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:

    model.eval()

    # tgt_vocab might be either a dict-like with .itos list, or expose lookup_token()
    def idx_to_tok(idx: int) -> str:
        return tgt_vocab[idx]

    PAD_IDX, SOS_IDX, EOS_IDX = 1, 2, 3

    def ids_to_text(ids: list[int]) -> str:
        toks = []
        for i in ids:
            if i in (PAD_IDX, SOS_IDX):
                continue
            if i == EOS_IDX:
                break
            toks.append(idx_to_tok(i))
        return " ".join(toks)

    hypotheses, references = [], []

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            # decode each sentence in the batch independently
            for i in range(src.size(0)):
                src_i      = src[i:i+1]                                  # [1, src_len]
                src_mask_i = make_src_mask(src_i, pad_idx=PAD_IDX)

                pred_ids = greedy_decode(
                    model, src_i, src_mask_i, max_len,
                    start_symbol=SOS_IDX, end_symbol=EOS_IDX, device=device,
                )                                                        # [1, out_len]

                hyp = ids_to_text(pred_ids.squeeze(0).tolist())
                ref = ids_to_text(tgt[i].tolist())

                hypotheses.append(hyp)
                references.append(ref)

    bleu = corpus_bleu(hypotheses, [references])
    return bleu.score


def save_checkpoint(
    model, optimizer, scheduler, epoch,
    path: str = "checkpoint.pt",
    dataset = None,
) -> None:
    enc_layer = model.encoder.layers[0]
    model_config = {
    "src_vocab_size": model.src_embed.num_embeddings,
    "tgt_vocab_size": model.tgt_embed.num_embeddings,
    "d_model":        model.d_model,
    "N":              len(model.encoder.layers),
    "num_heads":      enc_layer.self_attn.num_heads,
    "d_ff":           enc_layer.ffn.linear1.out_features,
    "dropout":        enc_layer.dropout.p,
}

    ckpt = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "model_config":         model_config,
    }
    if dataset is not None:
        ckpt["src_vocab"] = dataset.src_vocab
        ckpt["src_itos"]  = dataset.src_itos
        ckpt["tgt_vocab"] = dataset.tgt_vocab
        ckpt["tgt_itos"]  = dataset.tgt_itos
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt["epoch"]

DEFAULT_CONFIG = {
    # data
    "batch_size":   64,
    "num_workers":  0,
    # model
    "d_model":      256,         # 512 is paper-standard; 256 trains faster on small GPUs
    "N":            3,           # paper uses 6; 3 is fine for Multi30k
    "num_heads":    8,
    "d_ff":         1024,
    "dropout":      0.1,
    "use_scaling":  True,
    "learned_pos":  False,
    # optim
    "use_noam":     True,        # False → fixed lr (Exp 2.1)
    "fixed_lr":     1e-4,
    "warmup_steps": 4000,
    "betas":        (0.9, 0.98),
    "eps":          1e-9,
    # loss
    "label_smoothing": 0.1,      # 0.0 → standard CE (Exp 2.5)
    # training
    "num_epochs":   20,
    "max_len":      100,
    # logging
    "wandb_project": "da6401-a3",
    "wandb_run_name": None,
    "wandb_mode":    "online",   # "disabled" to run without wandb
    "checkpoint_dir": "checkpoints",
}


def run_training_experiment(config: Optional[dict] = None) -> None:
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # ── 1. W&B init ──────────────────────────────────────────────────
    wandb.init(
        project=cfg["wandb_project"],
        name=cfg["wandb_run_name"],
        config=cfg,
        mode=cfg["wandb_mode"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # ── 2. Dataset + vocab ───────────────────────────────────────────
    print("Loading data…")
    train_ds = Multi30kDataset(split="train")
    train_ds.build_vocab()

    val_ds = Multi30kDataset(split="validation")
    val_ds.src_vocab, val_ds.src_itos = train_ds.src_vocab, train_ds.src_itos
    val_ds.tgt_vocab, val_ds.tgt_itos = train_ds.tgt_vocab, train_ds.tgt_itos

    test_ds = Multi30kDataset(split="test")
    test_ds.src_vocab, test_ds.src_itos = train_ds.src_vocab, train_ds.src_itos
    test_ds.tgt_vocab, test_ds.tgt_itos = train_ds.tgt_vocab, train_ds.tgt_itos

    src_vocab_size = len(train_ds.src_itos)
    tgt_vocab_size = len(train_ds.tgt_itos)
    print(f"vocab — src: {src_vocab_size}, tgt: {tgt_vocab_size}")

    # ── 3. DataLoaders ───────────────────────────────────────────────
    train_pairs = train_ds.process_data()
    val_pairs   = val_ds.process_data()
    test_pairs  = test_ds.process_data()

    train_loader = DataLoader(_PairsDataset(train_pairs),
                            batch_size=cfg["batch_size"], shuffle=True,
                            num_workers=cfg["num_workers"], collate_fn=_collate)
    val_loader   = DataLoader(_PairsDataset(val_pairs),
                            batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], collate_fn=_collate)
    test_loader  = DataLoader(_PairsDataset(test_pairs),
                          batch_size=cfg["batch_size"], shuffle=False,
                          num_workers=cfg["num_workers"], collate_fn=_collate)


    # ── 4. Model ─────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size = src_vocab_size,
        tgt_vocab_size = tgt_vocab_size,
        d_model        = cfg["d_model"],
        N              = cfg["N"],
        num_heads      = cfg["num_heads"],
        d_ff           = cfg["d_ff"],
        dropout        = cfg["dropout"],
        use_scaling    = cfg["use_scaling"],  
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params/1e6:.2f} M")
    wandb.watch(model, log="gradients", log_freq=100)   # for Exp 2.2 grad-norm tracking

    # ── 5. Optimizer ─────────────────────────────────────────────────
    base_lr = 0.5 if cfg["use_noam"] else cfg["fixed_lr"]
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=base_lr, betas=cfg["betas"], eps=cfg["eps"])

    # ── 6. Scheduler ─────────────────────────────────────────────────
    scheduler = (
        NoamScheduler(optimizer, d_model=cfg["d_model"], warmup_steps=cfg["warmup_steps"])
        if cfg["use_noam"] else None
    )

    # ── 7. Loss ──────────────────────────────────────────────────────
    if cfg["label_smoothing"] > 0:
        loss_fn = LabelSmoothingLoss(tgt_vocab_size, PAD_IDX, smoothing=cfg["label_smoothing"])
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    loss_fn = loss_fn.to(device)

    # build a unique filename from the run's hyperparams so concurrent/sequential runs don't clobber
    run_tag = (
        f"{cfg.get('wandb_run_name') or 'run'}"
        f"_dm{cfg['d_model']}_N{cfg['N']}_h{cfg['num_heads']}"
        f"_dff{cfg['d_ff']}_drop{cfg['dropout']}"
        f"_lr{'noam' if cfg['use_noam'] else cfg['fixed_lr']}"
        f"_ls{cfg['label_smoothing']}"
    )
    ckpt_path = os.path.join(cfg["checkpoint_dir"], f"{run_tag}.pt")

    global_step    = 0
    LOG_GRAD_STEPS = 1000

    best_val_loss = float("inf")
    for epoch in range(cfg["num_epochs"]):
        # ── manual train pass so we can log grad norms ──
        model.train()
        total_loss, total_tokens = 0.0, 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            src_mask = make_src_mask(src, pad_idx=PAD_IDX)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx=PAD_IDX)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            loss   = loss_fn(logits.reshape(-1, logits.size(-1)),
                            tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()

            # ── log Q/K grad norms in the first 1000 optimizer steps ──
            if global_step < LOG_GRAD_STEPS:
                q_norms = [layer.self_attn.W_Q.weight.grad.norm().item()
                        for layer in model.encoder.layers]
                k_norms = [layer.self_attn.W_K.weight.grad.norm().item()
                        for layer in model.encoder.layers]
                wandb.log({
                    "step":          global_step,
                    "q_grad_norm":   sum(q_norms) / len(q_norms),
                    "k_grad_norm":   sum(k_norms) / len(k_norms),
                    "q_grad_norm_l0": q_norms[0],
                    "k_grad_norm_l0": k_norms[0],
                })

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            n_tok = (tgt_out != PAD_IDX).sum().item()
            total_loss   += loss.item() * n_tok
            total_tokens += n_tok
            global_step  += 1

        train_loss = total_loss / max(total_tokens, 1)
        print(f"[epoch {epoch}] train loss = {train_loss:.4f}")

        val_loss = run_epoch(val_loader, model, loss_fn, None, None,
                            epoch, is_train=False, device=device)

        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         optimizer.param_groups[0]["lr"],
        }, step=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch,
                            path=ckpt_path, dataset=train_ds)
            print(f"  ↳ new best val_loss = {val_loss:.4f}  →  {ckpt_path}")



    # ── 9. Final BLEU on test set ────────────────────────────────────
    print("Evaluating BLEU on test set…")
    load_checkpoint(ckpt_path, model)
    bleu = evaluate_bleu(model, test_loader, train_ds.tgt_itos,
                        device=device, max_len=cfg["max_len"])
    wandb.log({"test_bleu": bleu})
    print(f"test BLEU = {bleu:.2f}")



    wandb.finish()



if __name__ == "__main__":
    run_training_experiment()
