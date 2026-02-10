import argparse
import os
import sys
import math
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import clip

from datasets import build_dataset, build_dataloaders
from fs.utils import attach_expert_metadata
from fs.utils.eval_utils import evaluate
from lorasquaredlib import (
    apply_lorasquared,
    load_lorasquared,
    set_router_state_for_layers,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LoRA^2 adapters on base and novel splits.")

    # data
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--root_path", required=True, type=str)
    parser.add_argument("--shots", required=True, type=int)
    parser.add_argument("--setting", default="base2new", choices=["base2new", "standard"])
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--seed", default=1, type=int)

    # model / lora^2 config (must match training)
    parser.add_argument("--backbone", default="ViT-B/16", type=str)
    parser.add_argument("--encoder", default="both", choices=["text", "vision", "both"])
    parser.add_argument("--position", default="all", type=str)
    parser.add_argument("--params", nargs="+", default=["q", "k", "v"])
    parser.add_argument("--lora_shared_rank", required=True, type=int)
    parser.add_argument("--lora_expert_rank", required=True, type=int)
    parser.add_argument("--lora_num_experts", type=int, default=0, help="Needed when lora_expert_assignment=random_balanced")
    parser.add_argument(
        "--lora_expert_assignment",
        type=str,
        default="per_class",
        choices=["per_class", "random_balanced"],
    )
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--dropout_rate", default=0.25, type=float)

    # router settings
    parser.add_argument(
        "--router_mode",
        type=str,
        default="ste_softmax",
        choices=["weighted", "gumbel", "ste", "ste_softmax"],
    )
    parser.add_argument("--router_temperature", type=float, default=1.0)

    parser.add_argument(
        "--entropy_threshold",
        type=float,
        default=None,
        help="If set, per-sample decision: high router entropy -> shared-only; low entropy -> router-weighted.",
    )

    # weights
    parser.add_argument("--adapter_path", required=True, type=str, help="Path to saved LoRA^2 adapters (.pt).")

    return parser.parse_args()


def main():
    args = parse_args()

    # load CLIP
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    logit_scale = 100

    # dataset and loaders
    dataset = build_dataset(
        dataset=args.dataset,
        root_path=args.root_path,
        shots=args.shots,
        setting=args.setting,
        seed=args.seed,
    )
    attach_expert_metadata(
        dataset,
        mode=args.lora_expert_assignment,
        num_experts=args.lora_num_experts if args.lora_expert_assignment != "per_class" else None,
        seed=args.seed,
    )
    _, val_loader, test_loader = build_dataloaders(args, dataset, preprocess)

    # determine num experts
    if args.lora_expert_assignment == "random_balanced":
        n_experts = args.lora_num_experts
    else:
        n_experts = len(dataset.classnames)

    # instrument model with LoRA^2 (router enabled)
    list_lora_layers = apply_lorasquared(
        clip_model,
        backbone=args.backbone,
        encoder=args.encoder,
        position=args.position,
        params=args.params,
        r_shared=args.lora_shared_rank,
        r_expert=args.lora_expert_rank,
        n_experts=n_experts,
        alpha_shared=args.alpha,
        alpha_expert=args.alpha,
        dropout_rate=args.dropout_rate,
        enable_router=True,
        router_temperature=args.router_temperature,
        router_mode=args.router_mode,
        verbose=False,
    )
    clip_model._lorasquared_layers = list_lora_layers

    # load adapters (shared + experts + router)
    if not os.path.isfile(args.adapter_path):
        raise FileNotFoundError(f"Adapter file not found: {args.adapter_path}")
    missing, unexpected = load_lorasquared(
        clip_model,
        args.adapter_path,
        include_router=True,
        map_location="cpu",
        strict=False,
    )
    if missing:
        print(f"[load] Missing keys: {missing}")
    if unexpected:
        print(f"[load] Unexpected keys: {unexpected}")

    clip_model = clip_model.cuda().float()

    if args.setting == "base2new":
        test_base_loader, test_new_loader = test_loader

        acc_test_base = evaluate_entropy_routed(
            clip_model,
            test_base_loader,
            dataset,
            use_entropy=args.entropy_threshold is not None,
            entropy_threshold=args.entropy_threshold or 0.0,
            temperature=args.router_temperature,
            router_layers=list_lora_layers,
        )

        # novel: shared only
        set_router_state_for_layers(list_lora_layers, False)
        acc_test_novel = evaluate(
            clip_model,
            test_new_loader,
            template=dataset.template[0],
            classnames=dataset.test_new_classnames,
            use_expert=False,
        )
        set_router_state_for_layers(list_lora_layers, True)

        print(f"Test-Base (router/entropy): {acc_test_base:.2f}")
        print(f"Test-Novel (shared only): {acc_test_novel:.2f}")
    else:
        # standard setting: evaluate all classes; router ON with optional entropy routing
        acc_test = evaluate_entropy_routed(
            clip_model,
            test_loader,
            dataset,
            use_entropy=args.entropy_threshold is not None,
            entropy_threshold=args.entropy_threshold or 0.0,
            temperature=args.router_temperature,
            router_layers=list_lora_layers,
        )
        print(f"Test accuracy (router/entropy): {acc_test:.2f}")


@torch.no_grad()
def evaluate_entropy_routed(
    clip_model,
    loader,
    dataset,
    use_entropy: bool,
    entropy_threshold: float,
    temperature: float,
    router_layers,
):
    """
    If use_entropy:
        - Compute router weights entropy (from first router hook).
        - High entropy -> shared-only logits.
        - Low entropy -> router-enabled logits.
    Otherwise: router-enabled for all samples.
    """
    clip_model.eval()
    set_router_state_for_layers(router_layers, True)

    # Precompute text features (shared only to avoid routing in text encoder)
    set_router_state_for_layers(router_layers, False)
    texts = [dataset.template[0].format(c.replace("_", " ")) for c in dataset.classnames]
    tokenized = clip.tokenize(texts).cuda()
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        class_embeddings = clip_model.encode_text(tokenized)
    text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
    set_router_state_for_layers(router_layers, True)

    acc = 0.0
    tot = 0

    capture = {"logits": None}

    def router_hook(module, inp, out):
        capture["logits"] = out.detach()

    hook_handle = None
    for layer in router_layers:
        if hasattr(layer, "q_proj") and hasattr(layer.q_proj, "router") and layer.q_proj.router is not None:
            hook_handle = layer.q_proj.router.register_forward_hook(router_hook)
            break
    if hook_handle is None:
        raise RuntimeError("No router found to compute entropy.")

    for images, target in loader:
        images, target = images.cuda(), target.cuda()

        # router-enabled pass
        capture["logits"] = None
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            img_router = clip_model.encode_image(images)
        img_router = img_router / img_router.norm(dim=-1, keepdim=True)
        logits_router = img_router @ text_features.t()

        if use_entropy and capture["logits"] is not None:
            logits = capture["logits"]  # [B*seq, n_experts]
            bsz = images.shape[0]
            seq_len = logits.shape[0] // bsz
            logits = logits.view(bsz, seq_len, -1)
            weights = torch.softmax(logits / temperature, dim=-1)
            entropy = -(weights * (weights.clamp_min(1e-9).log())).sum(dim=-1)  # [B, seq_len]
            entropy_per_sample = entropy.mean(dim=1)  # [B]
            high = entropy_per_sample > entropy_threshold

            # shared-only pass
            set_router_state_for_layers(router_layers, False)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                img_shared = clip_model.encode_image(images)
            img_shared = img_shared / img_shared.norm(dim=-1, keepdim=True)
            logits_shared = img_shared @ text_features.t()
            set_router_state_for_layers(router_layers, True)

            high = high.unsqueeze(1)
            logits_final = torch.where(high, logits_shared, logits_router)
        else:
            logits_final = logits_router

        acc += cls_acc(logits_final, target) * len(logits_final)
        tot += len(logits_final)

    if hook_handle:
        hook_handle.remove()

    return acc / tot
