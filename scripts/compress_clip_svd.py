import argparse
import os
import sys
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import clip
from datasets import build_dataset, build_dataloaders
from fs.utils import attach_expert_metadata
from fs.utils.eval_utils import evaluate


def parse_args():
    p = argparse.ArgumentParser(
        description="Compress vanilla CLIP with rank-k SVD per linear layer and evaluate base/novel splits."
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--root_path", required=True)
    p.add_argument("--shots", required=True, type=int)
    p.add_argument("--setting", default="base2new", choices=["base2new"])
    p.add_argument("--backbone", default="ViT-B/16")
    p.add_argument("--rank", type=int, default=8, help="Rank to keep in SVD approximation.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--test_batch_size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--results_csv", type=str, help="Optional path to append results CSV.")
    return p.parse_args()


def compress_linear(layer: torch.nn.Linear, k: int):
    W = layer.weight.data.float()  # ensure float32 for SVD
    device = layer.weight.data.device
    W_cpu = W.detach().cpu()
    try:
        U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
    except RuntimeError:
        U, S, Vh = torch.svd(W_cpu)
    k = min(k, S.numel())
    W_approx = (U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]).to(device)
    layer.weight.data.copy_(W_approx)
    # keep bias unchanged


def compress_model(model: torch.nn.Module, k: int):
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            compress_linear(module, k)


def maybe_save_csv(args, metrics):
    if not args.results_csv:
        return
    import pandas as pd

    os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)
    row = {"dataset": args.dataset, "setting": args.setting, "shots": args.shots, "seed": args.seed, "rank": args.rank}
    row.update(metrics)
    df = pd.DataFrame([row])
    exists = os.path.isfile(args.results_csv)
    df.to_csv(args.results_csv, mode="a", header=not exists, index=False)


def main():
    args = parse_args()
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    # compress all linear layers
    compress_model(clip_model, args.rank)

    dataset = build_dataset(
        dataset=args.dataset,
        root_path=args.root_path,
        shots=args.shots,
        setting=args.setting,
        seed=args.seed,
    )
    attach_expert_metadata(dataset, mode="per_class", num_experts=None, seed=args.seed)
    _, _, test_loader = build_dataloaders(args, dataset, preprocess)
    test_base_loader, test_new_loader = test_loader

    # evaluate shared-only (experts not used)
    acc_test_base = evaluate(
        clip_model,
        test_base_loader,
        template=dataset.template[0],
        classnames=dataset.test_classnames,
        use_expert=False,
    )
    acc_test_novel = evaluate(
        clip_model,
        test_new_loader,
        template=dataset.template[0],
        classnames=dataset.test_new_classnames,
        use_expert=False,
    )

    print(f"SVD rank-{args.rank} | Base accuracy: {acc_test_base:.2f}")
    print(f"SVD rank-{args.rank} | Novel accuracy: {acc_test_novel:.2f}")
    maybe_save_csv(args, {"acc_test_base": acc_test_base, "acc_test_novel": acc_test_novel})


if __name__ == "__main__":
    main()
