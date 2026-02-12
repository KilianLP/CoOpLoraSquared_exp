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
from fs.utils.eval_utils import evaluate, cls_acc
from lorasquaredlib import (
    apply_lorasquared,
    load_lorasquared,
    set_router_state_for_layers,
)
from loralib.utils import apply_lora, mark_only_lora_as_trainable


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
    parser.add_argument("--r", type=int, default=4, help="LoRA rank (for plain LoRA adapter).")
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
    parser.add_argument(
        "--dual_eval",
        action="store_true",
        help="Evaluate with both a LoRA adapter (base) and a LoRA^2 adapter (shared for novel) using entropy over image logits.",
    )
    parser.add_argument("--lora_adapter_path", type=str, help="LoRA adapter weights (.pt) for dual_eval.")
    parser.add_argument("--lorasq_adapter_path", type=str, help="LoRA^2 adapter weights (.pt) for dual_eval.")
    parser.add_argument(
        "--image_entropy_choice",
        action="store_true",
        help="Select per-sample between router-on and shared-only image logits based on lower entropy.",
    )

    # weights
    parser.add_argument("--adapter_path", type=str, help="Path to saved LoRA^2 adapters (.pt).")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.dual_eval:
        run_dual_eval(args)
        return
    if not args.adapter_path:
        raise ValueError("--adapter_path is required unless --dual_eval is used.")

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

        if args.image_entropy_choice:
            acc_test_base = evaluate_image_entropy_choice(
                clip_model,
                test_base_loader,
                dataset,
                router_layers=list_lora_layers,
                text_router_on=True,
                temperature=args.router_temperature,
            )
            acc_test_novel = evaluate_image_entropy_choice(
                clip_model,
                test_new_loader,
                dataset,
                router_layers=list_lora_layers,
                text_router_on=False,  # novel text shared-only
                temperature=args.router_temperature,
            )
        else:
            acc_test_base = evaluate_entropy_routed(
                clip_model,
                test_base_loader,
                dataset,
                use_entropy=args.entropy_threshold is not None,
                entropy_threshold=args.entropy_threshold or 0.0,
                temperature=args.router_temperature,
                router_layers=list_lora_layers,
            )
            set_router_state_for_layers(list_lora_layers, False)
            acc_test_novel = evaluate(
                clip_model,
                test_new_loader,
                template=dataset.template[0],
                classnames=dataset.test_new_classnames,
                use_expert=False,
            )
            set_router_state_for_layers(list_lora_layers, True)

        print(f"Test-Base (router/entropy image choice): {acc_test_base:.2f}")
        print(f"Test-Novel (router/entropy image choice, text shared): {acc_test_novel:.2f}")
    else:
        # standard setting: evaluate all classes; router ON with optional entropy routing
        if args.image_entropy_choice:
            acc_test = evaluate_image_entropy_choice(
                clip_model,
                test_loader,
                dataset,
                router_layers=list_lora_layers,
                text_router_on=True,
                temperature=args.router_temperature,
            )
        else:
            acc_test = evaluate_entropy_routed(
                clip_model,
                test_loader,
                dataset,
                use_entropy=args.entropy_threshold is not None,
                entropy_threshold=args.entropy_threshold or 0.0,
                temperature=args.router_temperature,
                router_layers=list_lora_layers,
            )
        print(f"Test accuracy (router/entropy or image-choice): {acc_test:.2f}")


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


def _load_lora_from_path(path, list_lora_layers):
    if not os.path.exists(path):
        raise FileNotFoundError(f"LoRA adapter not found at {path}")
    data = torch.load(path, map_location="cpu")
    weights = data.get("weights", data)
    for i, layer in enumerate(list_lora_layers):
        layer_key = f"layer_{i}"
        if layer_key not in weights:
            continue
        w = weights[layer_key]
        if "q_proj" in w:
            layer.q_proj.w_lora_A.data.copy_(w["q_proj"]["w_lora_A"])
            layer.q_proj.w_lora_B.data.copy_(w["q_proj"]["w_lora_B"])
        if "k_proj" in w:
            layer.k_proj.w_lora_A.data.copy_(w["k_proj"]["w_lora_A"])
            layer.k_proj.w_lora_B.data.copy_(w["k_proj"]["w_lora_B"])
        if "v_proj" in w:
            layer.v_proj.w_lora_A.data.copy_(w["v_proj"]["w_lora_A"])
            layer.v_proj.w_lora_B.data.copy_(w["v_proj"]["w_lora_B"])
        if "proj" in w:
            layer.proj.w_lora_A.data.copy_(w["proj"]["w_lora_A"])
            layer.proj.w_lora_B.data.copy_(w["proj"]["w_lora_B"])


def run_dual_eval(args):
    """
    Dual evaluation:
      - LoRA adapter (base) loaded on model A.
      - LoRA^2 adapter (shared/expert) loaded on model B (router disabled for novel text; shared-only image pass option).
      - Base classes: text from LoRA; images: compare logits from LoRA image vs LoRA^2-shared image, pick lower-entropy.
      - Novel classes: text from LoRA^2 shared; images: same entropy choice between LoRA image and LoRA^2 shared image.
    """
    if not args.lora_adapter_path or not args.lorasq_adapter_path:
        raise ValueError("--dual_eval requires --lora_adapter_path and --lorasq_adapter_path")

    # CLIP models
    clip_lora, preprocess = clip.load(args.backbone)
    clip_lora.eval()
    clip_sq, _ = clip.load(args.backbone)
    clip_sq.eval()

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
    _, _, test_loader = build_dataloaders(args, dataset, preprocess)
    if args.setting == "base2new":
        test_base_loader, test_new_loader = test_loader
    else:
        test_base_loader = test_loader
        test_new_loader = None

    # LoRA (model A)
    list_lora_layers = apply_lora(args, clip_lora, verbose=False)
    mark_only_lora_as_trainable(clip_lora)
    _load_lora_from_path(args.lora_adapter_path, list_lora_layers)

    # LoRA^2 (model B)
    n_experts = args.lora_num_experts if args.lora_expert_assignment == "random_balanced" else len(dataset.classnames)
    lorasq_layers = apply_lorasquared(
        clip_sq,
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
        enable_router=False,
        verbose=False,
    )
    clip_sq._lorasquared_layers = lorasq_layers
    missing, unexpected = load_lorasquared(
        clip_sq,
        args.lorasq_adapter_path,
        include_router=False,
        map_location="cpu",
        strict=False,
    )
    if missing:
        print(f"[LoRA^2 load] Missing keys: {missing}")
    if unexpected:
        print(f"[LoRA^2 load] Unexpected keys: {unexpected}")

    clip_lora = clip_lora.cuda().float()
    clip_sq = clip_sq.cuda().float()

    def encode_text(model, classnames):
        texts = [dataset.template[0].format(c.replace("_", " ")) for c in classnames]
        tokenized = clip.tokenize(texts).cuda()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            class_embeddings = model.encode_text(tokenized)
        return class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    # Base text from LoRA, novel text from LoRA^2 shared
    text_base = encode_text(clip_lora, dataset.test_classnames if args.setting == "base2new" else dataset.classnames)
    text_novel = None
    if test_new_loader is not None:
        text_novel = encode_text(clip_sq, dataset.test_new_classnames)

    def image_entropy_choice(images, text_feats):
        # logits from LoRA
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            img_lora = clip_lora.encode_image(images)
        img_lora = img_lora / img_lora.norm(dim=-1, keepdim=True)
        logits_lora = img_lora @ text_feats.t()

        # logits from shared-only LoRA^2 (router off)
        set_router_state_for_layers(lorasq_layers, False)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            img_shared = clip_sq.encode_image(images)
        img_shared = img_shared / img_shared.norm(dim=-1, keepdim=True)
        logits_shared = img_shared @ text_feats.t()
        set_router_state_for_layers(lorasq_layers, False)

        def ent(logits):
            p = torch.softmax(logits, dim=-1)
            return -(p * p.clamp_min(1e-9).log()).sum(dim=-1)

        ent_l = ent(logits_lora)
        ent_s = ent(logits_shared)
        use_shared = ent_s < ent_l
        logits_final = torch.where(use_shared.unsqueeze(1), logits_shared, logits_lora)
        return logits_final

    def eval_loader(loader, text_feats):
        acc = 0.0
        tot = 0
        for images, target in loader:
            images, target = images.cuda(), target.cuda()
            logits = image_entropy_choice(images, text_feats)
            acc += cls_acc(logits, target) * len(logits)
            tot += len(logits)
        return acc / tot

    if args.setting == "base2new":
        acc_base = eval_loader(test_base_loader, text_base)
        acc_novel = eval_loader(test_new_loader, text_novel)
        print(f"Dual eval - Base (LoRA text, entropy img choice): {acc_base:.2f}")
        print(f"Dual eval - Novel (shared text, entropy img choice): {acc_novel:.2f}")
    else:
        acc = eval_loader(test_base_loader, text_base)
        print(f"Dual eval - Standard (entropy img choice): {acc:.2f}")

@torch.no_grad()
def evaluate_image_entropy_choice(
    clip_model,
    loader,
    dataset,
    router_layers,
    text_router_on: bool,
    temperature: float,
):
    """
    Text encoder: router on for base (text_router_on=True) else shared only.
    Image encoder: compute logits with router ON and shared-only; pick per-sample lower-entropy distribution.
    """
    clip_model.eval()

    # Text features
    set_router_state_for_layers(router_layers, text_router_on)
    texts = [dataset.template[0].format(c.replace("_", " ")) for c in dataset.classnames]
    tokenized = clip.tokenize(texts).cuda()
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        class_embeddings = clip_model.encode_text(tokenized)
    text_features = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

    acc = 0.0
    tot = 0

    for images, target in loader:
        images, target = images.cuda(), target.cuda()

        # Router ON pass
        set_router_state_for_layers(router_layers, True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            img_router = clip_model.encode_image(images)
        img_router = img_router / img_router.norm(dim=-1, keepdim=True)
        logits_router = img_router @ text_features.t()

        # Shared-only pass
        set_router_state_for_layers(router_layers, False)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            img_shared = clip_model.encode_image(images)
        img_shared = img_shared / img_shared.norm(dim=-1, keepdim=True)
        logits_shared = img_shared @ text_features.t()
        set_router_state_for_layers(router_layers, True)

        # Entropy per sample
        def entropy_from_logits(lgts):
            probs = torch.softmax(lgts / temperature, dim=-1)
            return -(probs * (probs.clamp_min(1e-9).log())).sum(dim=-1)

        ent_router = entropy_from_logits(logits_router)
        ent_shared = entropy_from_logits(logits_shared)

        use_shared = ent_shared < ent_router
        use_shared = use_shared.unsqueeze(1)
        logits_final = torch.where(use_shared, logits_shared, logits_router)

        acc += cls_acc(logits_final, target) * len(logits_final)
        tot += len(logits_final)

    return acc / tot


if __name__ == "__main__":
    main()
