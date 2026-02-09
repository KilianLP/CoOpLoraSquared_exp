import clip
import torch
import torch.nn.functional as F
import json
import os

from lorasquaredlib import (
    apply_lorasquared,
    mark_only_lorasquared_as_trainable,
    get_lorasquared_parameters,
    set_active_expert_for_layers,
    set_average_expert_mode_for_layers,
    shared_expert_orthogonality_loss,
)
from fs.utils.eval_utils import clip_classifier, cls_acc, evaluate


def _set_active_expert(layers, expert_selection):
    set_active_expert_for_layers(layers, expert_selection)


def _validate_expert_config(args, dataset, n_experts: int):
    if not hasattr(dataset, "label_to_expert_train"):
        raise ValueError(
            "Dataset missing expert metadata for LoRA^2. "
            "Ensure attach_expert_metadata() is called."
        )
    if getattr(dataset, "label_to_expert_train", None) is None:
        raise ValueError(
            "Dataset does not provide per-class expert assignments. "
            "Please run attach_expert_metadata(dataset) before training."
        )
    if args.lora_expert_rank <= 0:
        raise ValueError("LoRA^2 requires --lora_expert_rank > 0.")
    if n_experts <= 0:
        raise ValueError("LoRA^2 requires at least one expert.")
    train_lookup = dataset.label_to_expert_train
    if train_lookup.max().item() >= n_experts:
        raise ValueError(
            "Number of experts must cover all base (training) classes. "
            f"Got {n_experts}, but encountered expert id "
            f"{int(train_lookup.max().item())}."
        )


def _experts_for_targets(
    targets: torch.Tensor, class_expert_lookup: torch.Tensor
) -> torch.Tensor:
    """
    Map class labels to their assigned expert ids.
    """
    if targets.dtype != torch.long:
        targets = targets.long()
    return class_expert_lookup.index_select(0, targets)


def _safe_label_map(tensor: torch.Tensor | None, n_experts: int):
    if tensor is None:
        return None
    if tensor.numel() == 0:
        return None
    if tensor.max().item() >= n_experts:
        return None
    return tensor


def _base_eval_config(mode: str, mapping: torch.Tensor | None, n_experts: int):
    if mode == "shared":
        return None, False, None, False
    if mode == "avg_experts":
        return None, False, list(range(n_experts)), True
    return _safe_label_map(mapping, n_experts), True, None, False


def run_lorasquared(
    args,
    clip_model,
    logit_scale,
    dataset,
    train_loader,
    val_loader,
    test_loader,
):
    validate = getattr(args, "validate", False)
    dynamic_eval = getattr(args, "dynamic_eval", False)
    base_classnames = getattr(dataset, "classnames", None)
    if base_classnames is None or len(base_classnames) == 0:
        raise ValueError(
            "Dataset must expose `classnames` for the base split to size the expert pool."
        )
    assignment_mode = getattr(args, "lora_expert_assignment", "per_class")
    if assignment_mode == "random_balanced":
        n_experts = args.lora_num_experts
        if n_experts is None or n_experts <= 0:
            raise ValueError("random_balanced mode requires --lora_num_experts > 0")
    else:
        n_experts = len(base_classnames)
    _validate_expert_config(args, dataset, n_experts)

    # textual features of the training set
    textual_features = clip_classifier(
        dataset.classnames, dataset.template, clip_model
    )

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
        verbose=False,
    )
    clip_model._lorasquared_layers = list_lora_layers
    _set_active_expert(list_lora_layers, None)
    mark_only_lorasquared_as_trainable(clip_model)
    trainable_params = get_lorasquared_parameters(clip_model)

    clip_model = clip_model.cuda().float()
    base_eval_mode = getattr(args, "lorasquared_base_eval", "experts")
    dynamic_eval_enabled = dynamic_eval and base_eval_mode == "shared"
    if dynamic_eval and not dynamic_eval_enabled:
        print("Dynamic evaluation is only supported for --lorasquared_base_eval shared; disabling dynamic eval.")
    dynamic_eval_records = []
    class_expert_lookup = dataset.label_to_expert_train.to(
        clip_model.logit_scale.device
    )
    total_iters = args.n_iters * args.shots
    if args.setting == "base2new":
        test_base_loader, test_new_loader = test_loader
    else:
        test_base_loader = test_loader
        test_new_loader = None

    optimizer = torch.optim.AdamW(
        trainable_params,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_iters, eta_min=1e-6
    )

    scaler = torch.amp.GradScaler("cuda")
    count_iters = 0

    while count_iters < total_iters:
        clip_model.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.0
        loss_ortho_epoch = 0.0
        if args.encoder == "vision":
            text_features = textual_features.t().half()

        for i, (images, target) in enumerate(train_loader):
            template = dataset.template[0]
            texts = [
                template.format(classname.replace("_", " "))
                for classname in dataset.classnames
            ]
            images, target = images.cuda(), target.cuda()

            if args.encoder in ("text", "both"):
                _set_active_expert(list_lora_layers, class_expert_lookup)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    tokenized = clip.tokenize(texts).cuda()
                    class_embeddings = clip_model.encode_text(tokenized)
                text_features = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )

            sample_experts = _experts_for_targets(
                target, class_expert_lookup
            )
            _set_active_expert(list_lora_layers, sample_experts)

            if args.encoder in ("vision", "both"):
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model.encode_image(images)
            else:
                with torch.no_grad():
                    with torch.amp.autocast(
                        device_type="cuda", dtype=torch.float16
                    ):
                        image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )

            cosine_similarity = logit_scale * image_features @ text_features.t()
            loss_ce = F.cross_entropy(cosine_similarity, target)

            ortho_loss = 0.0
            if (
                args.lora_ortho_lambda > 0
                and args.lora_shared_rank > 0
                and args.lora_expert_rank > 0
            ):
                with torch.cuda.amp.autocast(enabled=False):
                    ortho_loss = shared_expert_orthogonality_loss(
                        list_lora_layers, eps=args.lora_ortho_eps
                    )
                loss = loss_ce + args.lora_ortho_lambda * ortho_loss
            else:
                loss = loss_ce

            acc_train += cls_acc(cosine_similarity, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            if isinstance(ortho_loss, torch.Tensor):
                loss_ortho_epoch += ortho_loss.item() * target.shape[0]
            tot_samples += target.shape[0]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()
            scheduler.step()

            count_iters += 1

            if dynamic_eval_enabled and count_iters % 25 == 0:
                clip_model.eval()
                if args.setting == "base2new":
                    acc_test_base = evaluate(
                        clip_model,
                        test_base_loader,
                        template=dataset.template[0],
                        classnames=dataset.test_classnames,
                        label_to_expert=None,
                        use_expert=False,
                    )
                    acc_test_novel = evaluate(
                        clip_model,
                        test_new_loader,
                        template=dataset.template[0],
                        classnames=dataset.test_new_classnames,
                        label_to_expert=None,
                        use_expert=False,
                    )
                    dynamic_eval_records.append(
                        {
                            "iteration": count_iters,
                            "acc_test_base": acc_test_base,
                            "acc_test_new": acc_test_novel,
                        }
                    )
                else:
                    acc_test = evaluate(
                        clip_model,
                        test_base_loader,
                        template=dataset.template[0],
                        classnames=dataset.test_classnames,
                        label_to_expert=None,
                        use_expert=False,
                    )
                    dynamic_eval_records.append(
                        {"iteration": count_iters, "acc_test": acc_test}
                    )
                clip_model.train()

            if count_iters == total_iters:
                break

            if args.debug and count_iters >= len(train_loader):
                count_iters = total_iters
                break

        if count_iters <= total_iters:
            acc_train /= tot_samples
            loss_epoch /= tot_samples
            ortho_epoch = loss_ortho_epoch / tot_samples if tot_samples > 0 else 0.0
            current_lr = scheduler.get_last_lr()[0]
            if args.lora_ortho_lambda > 0:
                print(
                    "[{}/{}] LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}, Ortho: {:.4f}".format(
                        count_iters, total_iters, current_lr, acc_train, loss_epoch, ortho_epoch
                    )
                )
            else:
                print(
                    "[{}/{}] LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}".format(
                        count_iters, total_iters, current_lr, acc_train, loss_epoch
                    )
                )

        if validate:
            clip_model.eval()
            val_mapping = _safe_label_map(dataset.label_to_expert_val, n_experts)
            acc_val = evaluate(
                clip_model,
                val_loader,
                template=dataset.template[0],
                classnames=dataset.val_classnames,
                label_to_expert=val_mapping,
            )
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))

    if args.save_path is not None:
        print("LoRA^2 saving is not implemented yet; skipping serialization.")

    _set_active_expert(list_lora_layers, None)

    if args.setting == "base2new":
        base_mapping, base_use_expert, base_override, avg_flag = _base_eval_config(
            base_eval_mode, dataset.label_to_expert_test, n_experts
        )
        if avg_flag:
            set_average_expert_mode_for_layers(list_lora_layers, True)
        acc_test_base = evaluate(
            clip_model,
            test_base_loader,
            template=dataset.template[0],
            classnames=dataset.test_classnames,
            label_to_expert=base_mapping,
            use_expert=base_use_expert,
            expert_override=base_override,
        )
        if avg_flag:
            set_average_expert_mode_for_layers(list_lora_layers, False)
        print("**** Test-Base accuracy: {:.2f}. ****\n".format(acc_test_base))

        # New classes should rely on the shared adapter only.
        acc_test_novel = evaluate(
            clip_model,
            test_new_loader,
            template=dataset.template[0],
            classnames=dataset.test_new_classnames,
            use_expert=False,
        )
        print("**** Test-Novel accuracy: {:.2f}. ****\n".format(acc_test_novel))
        result = {"acc_test_base": acc_test_base, "acc_test_new": acc_test_novel}

    else:
        test_mapping, test_use_expert, test_override, avg_flag = _base_eval_config(
            base_eval_mode, dataset.label_to_expert_test, n_experts
        )
        if avg_flag:
            set_average_expert_mode_for_layers(list_lora_layers, True)
        acc_test = evaluate(
            clip_model,
            test_loader,
            template=dataset.template[0],
            classnames=dataset.test_classnames,
            label_to_expert=test_mapping,
            use_expert=test_use_expert,
            expert_override=test_override,
        )
        if avg_flag:
            set_average_expert_mode_for_layers(list_lora_layers, False)
        print(
            "\n**** Final test accuracy (all categories): {:.2f}. ****\n".format(
                acc_test
            )
        )
        result = {"acc_test": acc_test}

    _write_dynamic_eval(args, dynamic_eval_records, base_eval_mode)
    return result


def _write_dynamic_eval(args, records, base_eval_mode):
    if not records:
        return
    backbone = args.backbone.replace("/", "-")
    mode_dir = args.mode
    if mode_dir == "lorasquared" and base_eval_mode:
        mode_dir = f"{mode_dir}_{base_eval_mode}"
    base_dir = os.path.join(
        args.results_dir,
        args.setting,
        backbone,
        args.dataset,
        f"shots_{args.shots}",
        f"seed_{args.seed}",
        mode_dir,
    )
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, f"{args.exp_name}_dynamic_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
