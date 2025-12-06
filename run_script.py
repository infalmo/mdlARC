from pathlib import Path
import argparse
import importlib
import utils, tinytransformer, train

importlib.reload(utils)  # pick up code changes during iteration
importlib.reload(tinytransformer)
importlib.reload(train)

args = {
    # run config
    "num_workers": 0,
    "device": "cuda",  # 'cuda' | 'mps' | 'cpu'
    # color augmentation
    "enable_color_aug_train": False,
    "max_color_augments_train": 0,
    "enable_color_aug_eval": False,
    "max_color_augments_eval": 0,
    "color_aug_seed": None,
    # visualization controls
    "visualize_augmented_outputs": False,
    "visualize_task_ids": [],  # e.g. ["00d62c1b"]
    "visualize_split": "test",
    "visualize_pair_index": None,  # None = all pairs/augments
    "visualize_plot": False,
    "visualize_log_prompts": False,
    "visualize_aaivr_top_k": 2,
    # paths - must pass as Path("<path_to_dir>")
    "save_path": Path("runs/tiny.pt"),
    "checkpoint_path": Path("runs/tiny.pt"),  # or None to start from scratch
    "data_path": Path(
        "assets/ARC-1/grouped-tasks/concept_plus_combined_dihedral_train/challenges.json"
    ),  # this dataset has dihedral augments only on the train set
    # hyperparameters
    "epochs": 1,
    "batch_size": 110,
    "val_batch_size": 60,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "seed": 42,
    # Visibility toggles
    "log_train_strings": False,
    "log_train_limit": 10,
    "log_inference_prompt": False,
}
cfg = argparse.Namespace(**args)

model, dataset, dataloader, device, data_path = train.build_model_and_data(cfg)

# Training only
train.train_model(
    cfg,
    model=model,
    dataloader=dataloader,
    dataset=dataset,
    device=device,
    data_path=data_path,
)

cfg.data_path = Path(
    "assets/ARC-1/grouped-tasks/concept_plus_combined_dihedral_both/challenges.json"
)  # note that for inference we switch to a dataset with dihedral augments on both train AND test to make AAIVR work
cfg.checkpoint_path = cfg.save_path

model, dataset, dataloader, device, data_path = train.build_model_and_data(
    cfg, is_eval=True
)

color_mappings_eval = None
color_apply_fn = None
if cfg.enable_color_aug_eval and cfg.max_color_augments_eval > 0:
    color_seed = cfg.color_aug_seed or cfg.seed
    color_mappings_eval = utils.generate_color_mapping_tensors(
        cfg.max_color_augments_eval, color_seed
    )
    color_apply_fn = lambda split: True

# # Full dataset evaluation
import inference

importlib.reload(inference)

EVAL_BATCH_SIZE = 1000
splits = ["train", "test"]

evaluation = inference.evaluate_model_on_dataset(
    model=model,
    dataset=dataset,
    device=device,
    batch_size=EVAL_BATCH_SIZE,
    log_prompts=args["log_inference_prompt"],
    splits=splits,
    color_mappings=color_mappings_eval,
    color_apply_fn=color_apply_fn,
)

for split in splits:
    summary = evaluation.get(split, {}).get("summary", {})
    total = summary.get("total_sequences", 0)
    shape_ok = summary.get("num_shape_correct", 0)
    avg_pixel_acc = summary.get("avg_pixel_accuracy", 0.0)
    fully_correct = summary.get("num_fully_correct", 0)

    print(f"\nSplit: {split}")
    print(f"  sequences evaluated: {total}")
    print(f"  correct output grid shapes: {shape_ok} / {total}")
    if shape_ok > 0:
        print(f"  avg pixel accuracy (shape-correct only): {avg_pixel_acc:.4f}")
    else:
        print("  avg pixel accuracy (shape-correct only): n/a")
    print(f"  fully correct output grids: {fully_correct} / {total}")

    if split == "test":
        correct_outputs = summary.get("fully_correct_results", [])
        print("  fully correct test outputs (task_id, pair_index, grid):")
        if not correct_outputs:
            print("    (none)")
        for res in correct_outputs:
            grid = res.get("output_grid", [])
            print(f"    - {res.get('task_id')} pair {res.get('pair_index')}: {grid}")

# # AAIVR voting on augmented test predictions
# import importlib
# import utils

importlib.reload(utils)

test_results = evaluation.get("test", {}).get("results", [])
aaivr_results = utils.run_aaivr_on_results(test_results)

print("\nAAIVR selections (pass@2) for test split:")
if not aaivr_results:
    print("  no test results available for AAIVR voting")

summary = utils.summarize_aaivr_pass_at_k(aaivr_results)
evaluated = summary.get("evaluated", 0)
hits = summary.get("hits", 0)

print("AAIVR pass@2 with targets:", f"{hits} / {evaluated} original test pairs")
x = set([])
for sel in aaivr_results:
    if sel.pass_at_k:
        x.add(sel.task_id)

print("Unique tasks: ", len(set(x)))

# Optional: Visualize augmented outputs for selected tasks (dihedral + color)
if cfg.visualize_augmented_outputs and cfg.visualize_task_ids:
    selected_split = cfg.visualize_split
    pair_idx = cfg.visualize_pair_index
    vis_plot = cfg.visualize_plot
    vis_log_prompts = cfg.visualize_log_prompts
    top_k = max(1, int(cfg.visualize_aaivr_top_k))

    split_results = evaluation.get(selected_split, {}).get("results", [])
    filtered = [
        res
        for res in split_results
        if res.get("task_id") in cfg.visualize_task_ids
        and (pair_idx is None or res.get("pair_index") == pair_idx)
    ]
    if not filtered:
        print(
            f"\n[visualize] No results found for tasks {cfg.visualize_task_ids} "
            f"in split '{selected_split}' (pair_index={pair_idx})."
        )
    else:
        pair_label = pair_idx if pair_idx is not None else "all"
        print(
            f"\n[visualize] Showing augmented predictions for tasks {cfg.visualize_task_ids} "
            f"in split '{selected_split}' (pair_index={pair_label})"
        )
        filtered.sort(
            key=lambda r: (
                r.get("task_id", ""),
                r.get("pair_index", -1),
                r.get("color_permutation_index", -1),
            )
        )
        for res in filtered:
            task_id = res.get("task_id")
            pair_index = res.get("pair_index")
            color_idx = res.get("color_permutation_index", None)
            color_label = f", color_perm={color_idx}" if color_idx is not None else ""
            print(f"\nTask {task_id} pair {pair_index} ({selected_split}{color_label})")
            if vis_log_prompts:
                print("Prompt tokens:", utils.tokens_to_string(res["prompt_tokens"]))
            print(
                "Generated output tokens:", utils.tokens_to_string(res["output_tokens"])
            )
            if res.get("target_output_tokens"):
                print(
                    "Target output tokens:",
                    utils.tokens_to_string(res["target_output_tokens"]),
                )
            print("Predicted grid:")
            for row in res["output_grid"]:
                print(row)
            if res.get("target_grid"):
                print("Target grid:")
                for row in res["target_grid"]:
                    print(row)
            if vis_plot:
                prompt_grids = utils.split_grids_from_tokens(res["prompt_tokens"])
                input_grid = prompt_grids[0] if prompt_grids else []
                to_plot = [input_grid, res["output_grid"]]
                if res.get("target_grid"):
                    to_plot.append(res["target_grid"])
                try:
                    utils.plot_grids(
                        to_plot,
                        title=(
                            f"{task_id} pair {pair_index} ({selected_split}{color_label})"
                        ),
                    )
                except Exception as e:
                    print(
                        f"  skipping visualization for {task_id} pair {pair_index}: {e}"
                    )

        if selected_split == "test":
            print(f"\n[visualize] AAIVR top-{top_k} for selected tasks")
            aaivr_subset = [res for res in filtered if res.get("split") == "test"]
            selections = utils.run_aaivr_on_results(
                aaivr_subset, top_k=top_k, discard_input_copies=True
            )
            if not selections:
                print("  no AAIVR selections for the chosen subset.")
            else:
                for sel in selections:
                    if sel.pass_at_k is None:
                        pass_str = "N/A"
                    else:
                        pass_str = "PASS" if sel.pass_at_k else "MISS"
                    print(
                        f"  Task {sel.task_id} base_pair {sel.original_pair_index}: "
                        f"{pass_str} (generated={sel.num_generated}, valid={sel.num_valid})"
                    )
                    if sel.target_grid is not None:
                        print("    Target grid:")
                        for row in sel.target_grid:
                            print(f"    {row}")
                    for idx, cand in enumerate(sel.ranked_candidates[:top_k]):
                        grid = cand["grid"]
                        count = cand["count"]
                        print(f"    Candidate {idx + 1} (count={count}):")
                        for row in grid:
                            print(f"      {row}")
                        if vis_plot:
                            try:
                                utils.plot_grids(
                                    [grid],
                                    title=(
                                        f"{sel.task_id} pair {sel.original_pair_index} "
                                        f"AAIVR cand {idx + 1}"
                                    ),
                                )
                            except Exception as e:
                                print(f"      plot failed: {e}")
