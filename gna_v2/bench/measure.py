"""
bench/measure.py — Measures what actually matters.
"""
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def throughput(model, seq_len, batch_size, device, n_warmup=5, n_runs=20):
    model.eval()
    x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    with torch.no_grad():
        for _ in range(n_warmup): model(x)
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs): model(x)
    if device.type == "cuda": torch.cuda.synchronize()
    return (batch_size * seq_len * n_runs) / (time.perf_counter() - t0)


def vram_mb(model, seq_len, batch_size, device):
    if device.type != "cuda": return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    model.eval()
    with torch.no_grad(): model(x)
    return torch.cuda.max_memory_allocated(device) / 1024**2


def train_one_epoch(model, dataset, batch_size, optimizer, device, scaler):
    model.train()
    total_loss, total_tokens = 0.0, 0
    t0 = time.time()
    for x, y in dataset.iter(batch_size):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            loss = F.cross_entropy(
                model(x).view(-1, model.decoder.weight.shape[0]), y.view(-1))
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        n = y.numel()
        total_loss   += loss.item() * n
        total_tokens += n
    return total_loss / total_tokens, total_tokens, time.time() - t0


@torch.no_grad()
def evaluate(model, dataset, batch_size):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in dataset.iter(batch_size, shuffle=False):
        loss = F.cross_entropy(
            model(x).view(-1, model.decoder.weight.shape[0]), y.view(-1))
        n = y.numel()
        total_loss   += loss.item() * n
        total_tokens += n
    return total_loss / total_tokens


def train_to_convergence(model, train_ds, valid_ds, batch_size, device,
                         lr=3e-4, max_epochs=40, patience=5, use_amp=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr/10)
    scaler = torch.amp.GradScaler() if use_amp and device.type == "cuda" else None

    best_val, no_improve = float("inf"), 0
    history = []
    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        train_loss, n_tok, train_s = train_one_epoch(
            model, train_ds, batch_size, optimizer, device, scaler)
        val_loss = evaluate(model, valid_ds, batch_size)
        scheduler.step()

        val_ppl   = math.exp(min(val_loss,   20))
        train_ppl = math.exp(min(train_loss, 20))
        tok_s     = n_tok / train_s

        improved = val_ppl < best_val
        if improved:
            best_val, no_improve = val_ppl, 0
        else:
            no_improve += 1

        history.append(dict(epoch=epoch, train_ppl=train_ppl, val_ppl=val_ppl,
                            tok_s=tok_s, train_s=train_s, improved=improved))

        mark = "✓" if improved else f"({no_improve}/{patience})"
        print(f"  ep {epoch:3d}  train {train_ppl:7.2f}  val {val_ppl:7.2f}  "
              f"best {best_val:7.2f}  {tok_s:,.0f} tok/s  {mark}")

        if no_improve >= patience:
            print(f"  Early stop.")
            break

    return dict(best_val_ppl=best_val, params=sum(p.numel() for p in model.parameters()
                if p.requires_grad), total_s=time.time()-t_start,
                n_epochs=len(history), history=history)
