"""
search/run.py — Exhaustive primitive search with timing and speed gating.
"""
import csv
import time
import torch
from gna_v2.blocks.model       import LM
from gna_v2.primitives.mixing  import MIXING
from gna_v2.primitives.channel import CHANNEL
from gna_v2.primitives.norm    import NORM
from gna_v2.bench.measure      import throughput, vram_mb, train_to_convergence
from gna_v2.data.ptb           import load


BASE_CFG = dict(
    dim        = 256,
    n_blocks   = 4,
    seq_len    = 128,
    dropout    = 0.1,
    n_heads    = 8,
    expand     = 4,
    rank       = 32,
    prenorm    = True,
    residual   = True,
)


def all_configs():
    for mix in MIXING:
        for chan in CHANNEL:
            for norm in NORM:
                yield dict(**BASE_CFG, mixing=mix, channel=chan, norm=norm)


def run_search(
    max_epochs:  int   = 20,
    patience:    int   = 5,
    batch_size:  int   = 128,
    out_csv:     str   = "search_results.csv",
    device:      torch.device = None,
    speed_gate:  float = 0.9,  # skip if slower than fastest * speed_gate
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = list(all_configs())
    n_total = len(configs)

    print(f"\nGNA V2 — Primitive Search")
    print(f"Device: {device}  |  Batch: {batch_size}  |  Epochs: {max_epochs}")
    print(f"Combos: {n_total}  |  Speed gate: skip if <{speed_gate*100:.0f}% of fastest\n")

    train_ds, valid_ds, vocab_sz = load(seq_len=BASE_CFG["seq_len"], device=device)

    rows        = []
    fields      = ["mixing", "channel", "norm", "params", "tok_per_s",
                   "vram_mb", "best_val_ppl", "total_s", "n_epochs", "skipped"]
    fastest_tps = None
    t_search    = time.time()
    n_done      = 0
    n_skipped   = 0

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for i, cfg in enumerate(configs):
            cfg["vocab_size"] = vocab_sz
            name = f"{cfg['mixing']:<18} | {cfg['channel']:<16} | {cfg['norm']}"

            elapsed     = time.time() - t_search
            avg_s       = elapsed / max(i, 1)
            remain_s    = avg_s * (n_total - i)
            eta         = f"ETA {remain_s/60:.0f}m" if i > 0 else "ETA ?"

            print(f"\n{'─'*60}")
            print(f"  [{i+1:3d}/{n_total}]  {eta}  done={n_done}  skipped={n_skipped}")
            print(f"  {name}")

            try:
                model  = LM(cfg).to(device)
                params = model.count_parameters()
                tps    = throughput(model, BASE_CFG["seq_len"], batch_size, device)
                vmb    = vram_mb(model, BASE_CFG["seq_len"], batch_size, device)

                if fastest_tps is None or tps > fastest_tps:
                    fastest_tps = tps

                threshold = fastest_tps * speed_gate
                if tps < threshold:
                    print(f"  SKIP  {tps:,.0f} tok/s < {threshold:,.0f} threshold")
                    row = dict(mixing=cfg["mixing"], channel=cfg["channel"],
                               norm=cfg["norm"], params=params,
                               tok_per_s=f"{tps:.0f}", vram_mb=f"{vmb:.1f}",
                               best_val_ppl=9999, total_s=0, n_epochs=0, skipped=1)
                    rows.append(row)
                    writer.writerow(row)
                    f.flush()
                    n_skipped += 1
                    continue

                print(f"  params={params:,}  tok/s={tps:,.0f}  vram={vmb:.0f}MB")

                result = train_to_convergence(
                    model, train_ds, valid_ds, batch_size, device,
                    max_epochs=max_epochs, patience=patience,
                )

                row = dict(
                    mixing       = cfg["mixing"],
                    channel      = cfg["channel"],
                    norm         = cfg["norm"],
                    params       = params,
                    tok_per_s    = f"{tps:.0f}",
                    vram_mb      = f"{vmb:.1f}",
                    best_val_ppl = f"{result['best_val_ppl']:.2f}",
                    total_s      = f"{result['total_s']:.1f}",
                    n_epochs     = result["n_epochs"],
                    skipped      = 0,
                )
                rows.append(row)
                writer.writerow(row)
                f.flush()
                n_done += 1

            except Exception as e:
                print(f"  ERROR: {e}")
                row = dict(mixing=cfg["mixing"], channel=cfg["channel"],
                           norm=cfg["norm"], params=0, tok_per_s=0,
                           vram_mb=0, best_val_ppl=9999, total_s=0, n_epochs=0, skipped=0)
                rows.append(row)
                writer.writerow(row)
                f.flush()

    total_elapsed = time.time() - t_search
    print(f"\n  Total: {total_elapsed/60:.1f} min  done={n_done}  skipped={n_skipped}")
    print_leaderboard(rows)
    return rows


def print_leaderboard(rows: list):
    valid = [r for r in rows if float(r["best_val_ppl"]) < 9999 and not int(r.get("skipped", 0))]
    if not valid:
        print("No valid results.")
        return

    print(f"\n{'='*80}")
    print("  LEADERBOARD — best val ppl")
    print(f"{'='*80}")
    print(f"  {'Mixing':<20} {'Channel':<18} {'Norm':<12} {'PPL':>7} {'Tok/s':>9} {'Params':>10}")
    print(f"  {'─'*76}")
    for r in sorted(valid, key=lambda x: float(x["best_val_ppl"])):
        print(f"  {r['mixing']:<20} {r['channel']:<18} {r['norm']:<12} "
              f"  {float(r['best_val_ppl']):6.1f} {float(r['tok_per_s']):9,.0f} "
              f"{int(r['params']):10,}")

    print(f"\n  FASTEST — top 5 tok/s")
    print(f"  {'─'*76}")
    for r in sorted(valid, key=lambda x: -float(x["tok_per_s"]))[:5]:
        print(f"  {r['mixing']:<20} {r['channel']:<18} {r['norm']:<12} "
              f"  {float(r['best_val_ppl']):6.1f} {float(r['tok_per_s']):9,.0f}")


if __name__ == "__main__":
    run_search()
