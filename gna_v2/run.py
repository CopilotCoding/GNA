"""
run.py — Entry point for GNA V2 research suite.

Usage:
    # Full search — all primitive combinations
    python run.py

    # Quick test — 5 epochs, fast combos only
    python run.py --quick

    # Single config test
    python run.py --mixing full_attention --channel mlp --norm layer_norm

    # Just profile throughput, no training
    python run.py --profile_only
"""
import argparse
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from gna_v2.search.run         import run_search, print_leaderboard, BASE_CFG
from gna_v2.blocks.model       import LM
from gna_v2.bench.measure      import throughput, vram_mb, train_to_convergence
from gna_v2.data.ptb           import load
from gna_v2.primitives.mixing  import MIXING
from gna_v2.primitives.channel import CHANNEL
from gna_v2.primitives.norm    import NORM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixing",  default=None, choices=list(MIXING))
    parser.add_argument("--channel", default=None, choices=list(CHANNEL))
    parser.add_argument("--norm",    default=None, choices=list(NORM))
    parser.add_argument("--epochs",  type=int, default=40)
    parser.add_argument("--patience",type=int, default=5)
    parser.add_argument("--batch",   type=int, default=128)
    parser.add_argument("--dim",     type=int, default=256)
    parser.add_argument("--blocks",  type=int, default=4)
    parser.add_argument("--out",     default="search_results.csv")
    parser.add_argument("--quick",   action="store_true",
                        help="5 epochs, dim=128 — fast sweep")
    parser.add_argument("--profile_only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nGNA V2 Research Suite")
    print(f"GPU: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")
    print(f"Available mixing:  {list(MIXING)}")
    print(f"Available channel: {list(CHANNEL)}")
    print(f"Available norm:    {list(NORM)}")

    if args.quick:
        args.epochs  = 5
        args.patience = 3
        args.dim     = 128
        args.blocks  = 3

    # Single config mode
    if args.mixing or args.channel or args.norm:
        cfg = dict(**BASE_CFG,
            mixing  = args.mixing  or "full_attention",
            channel = args.channel or "mlp",
            norm    = args.norm    or "layer_norm",
            dim     = args.dim,
            n_blocks= args.blocks,
        )
        train_ds, valid_ds, vocab_sz = load(device=device)
        cfg["vocab_size"] = vocab_sz

        model = LM(cfg).to(device)
        print(f"\nConfig: {cfg['mixing']} | {cfg['channel']} | {cfg['norm']}")
        print(f"Params: {model.count_parameters():,}")

        tps = throughput(model, 128, args.batch, device)
        vmb = vram_mb(model, 128, args.batch, device)
        print(f"Throughput: {tps:,.0f} tok/s  |  VRAM: {vmb:.0f} MB")

        if not args.profile_only:
            result = train_to_convergence(
                model, train_ds, valid_ds, args.batch, device,
                max_epochs=args.epochs, patience=args.patience,
            )
            print(f"\nBest val ppl: {result['best_val_ppl']:.2f}")
            print(f"Total time:   {result['total_s']/60:.1f} min")
        return

    # Full search
    run_search(
        max_epochs = args.epochs,
        patience   = args.patience,
        batch_size = args.batch,
        out_csv    = args.out,
        device     = device,
    )


if __name__ == "__main__":
    main()
