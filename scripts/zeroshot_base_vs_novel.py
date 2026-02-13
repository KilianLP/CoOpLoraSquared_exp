import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import clip
import torch

from datasets import build_dataset, build_dataloaders


def parse_args():
    p = argparse.ArgumentParser(
        description="Zero-shot base-vs-novel classifier using vanilla CLIP (no fine-tuning)."
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--root_path", required=True)
    p.add_argument("--shots", required=True, type=int)
    p.add_argument("--setting", default="base2new", choices=["base2new"])
    p.add_argument("--backbone", default="ViT-B/16")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--test_batch_size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


@torch.no_grad()
def encode_texts(clip_model, template, classnames):
    texts = [template[0].format(c.replace("_", " ")) for c in classnames]
    tokenized = clip.tokenize(texts).cuda()
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        embs = clip_model.encode_text(tokenized)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    return embs


@torch.no_grad()
def evaluate_split(loader, text_base, text_novel, clip_model):
    """Return accuracy of base-vs-novel decision for a given loader."""
    correct = 0
    total = 0
    for images, targets in loader:
        images, targets = images.cuda(), targets.cuda()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            img = clip_model.encode_image(images)
        img = img / img.norm(dim=-1, keepdim=True)

        logits_base = img @ text_base.t()
        logits_novel = img @ text_novel.t()
        max_base = logits_base.max(dim=1).values
        max_novel = logits_novel.max(dim=1).values

        # decide base vs novel per sample
        pred_is_base = max_base >= max_novel
        # ground truth: loader identity (base loader => base; novel loader => novel)
        gt_is_base = loader.is_base_split
        correct += (pred_is_base == gt_is_base).sum().item()
        total += pred_is_base.numel()
    return correct / total if total > 0 else 0.0


def main():
    args = parse_args()

    if args.setting != "base2new":
        raise ValueError("This script currently supports only base2new setting.")

    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    dataset = build_dataset(
        dataset=args.dataset,
        root_path=args.root_path,
        shots=args.shots,
        setting=args.setting,
        seed=args.seed,
    )
    _, _, test_loader = build_dataloaders(args, dataset, preprocess)
    test_base_loader, test_new_loader = test_loader
    # annotate loaders
    test_base_loader.is_base_split = True
    test_new_loader.is_base_split = False

    # text embeddings
    text_base = encode_texts(clip_model, dataset.template, dataset.test_classnames)
    text_novel = encode_texts(clip_model, dataset.template, dataset.test_new_classnames)

    acc_base = evaluate_split(test_base_loader, text_base, text_novel, clip_model)
    acc_novel = evaluate_split(test_new_loader, text_base, text_novel, clip_model)
    # overall
    total = acc_base * len(test_base_loader.dataset) + acc_novel * len(test_new_loader.dataset)
    overall = total / (len(test_base_loader.dataset) + len(test_new_loader.dataset))

    print(f"Base-vs-Novel accuracy (base split): {acc_base*100:.2f}")
    print(f"Base-vs-Novel accuracy (novel split): {acc_novel*100:.2f}")
    print(f"Base-vs-Novel accuracy (overall): {overall*100:.2f}")


if __name__ == "__main__":
    main()
