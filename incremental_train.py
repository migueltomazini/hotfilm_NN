"""Incremental / online training for hot-film neural network.

This script handles the sequential training logic, utilizing modular tools
from the utils package to execute dataset splits, fine-tuning, and evaluation.
"""

import os
import copy
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
import joblib
from sklearn.preprocessing import StandardScaler

from utils import config, data_loader, hyperparameter_optimization, model_utils
from train_mlp import MLP

input_size = config.INPUT_SIZE
output_size = config.OUTPUT_SIZE
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(
        description="Incremental training with block-wise metrics",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("serie", help="series identifier (e.g. 0610)")
    parser.add_argument("--num-blocks", dest="num_blocks", type=int, default=None, help="number of blocks to split the dataset into")
    parser.add_argument("--base-model", type=str, default=None, help="path to an existing .pth file for warm start")
    parser.add_argument("--scattered", action="store_true", help="Use scattered training mode")
    parser.add_argument("--percentage", type=float, default=100.0, help="Percentage of data to use from each block")
    parser.add_argument("--reverse-data", action="store_true", help="Inverts chronological order")
    parser.add_argument("--holdout-last", action="store_true", help="Reserves the final block strictly for testing")
    parser.add_argument("--no-finetune", action="store_true", help="[Test 1] Disables fine-tuning in scattered mode")
    parser.add_argument("--subsequent-finetune-pct", type=float, default=None, help="[Test 2] Percentage of block data to use for fine-tuning")
    
    args = parser.parse_args()
    NUM_BLOCKS = args.num_blocks
    SCATTERED = args.scattered
    PERCENTAGE = args.percentage
    serie = args.serie

    df_path = os.path.join(config.DATA_DIR, "train", f"train_df_{serie}.csv")
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Training data not found: {df_path}")
    df = pd.read_csv(df_path)

    if "reynolds" not in df.columns:
        cfg_path = os.path.join(config.DATA_DIR, "config", f"config_{serie}.json")
        re_val = 0.0
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as fh:
                    cfg_temp = json.load(fh)
                re_val = cfg_temp.get("RE_NUMBER", 0.0)
            except Exception:
                pass
        df["reynolds"] = re_val
        print(f"[Info] added missing reynolds column = {re_val}")

    with open(os.path.join(config.DATA_DIR, "config", f"config_{serie}.json")) as fh:
        cfg = json.load(fh)
    fs = cfg["FS_HOTFILM"]

    if args.reverse_data:
        print("\n" + "!"*70)
        print("🧪 [EXPERIMENTAL MODE] FLAG --reverse-data ACTIVATED!")
        df = df.iloc[::-1].reset_index(drop=True)
        serie = f"{serie}_rev"
        print("!"*70 + "\n")

    if NUM_BLOCKS is not None:
        print(f"Splitting into exactly {NUM_BLOCKS} blocks...")
        blocks = data_loader.split_dataframe_into_n_blocks(df, NUM_BLOCKS)
    else:
        blocks = data_loader.prepare_blocks(df, block_size=None, gap_threshold=None)

    if not SCATTERED:
        blocks = [b.iloc[: int(len(b) * PERCENTAGE / 100)].reset_index(drop=True) for b in blocks]

    if len(blocks) == 0:
        print("No blocks extracted from the dataset. Exiting.")
        return

    print(f"Dataset split into {len(blocks)} blocks")

    print("[Optimization] Optimizing hyperparameters...")
    if SCATTERED:
        opt_blocks = [block.iloc[: len(block) // len(blocks)] for block in blocks]
        opt_df = pd.concat(opt_blocks).reset_index(drop=True)
    else:
        opt_df = blocks[0].copy()

    X_opt = opt_df[["voltage_x", "voltage_y", "voltage_z", "reynolds"]].values
    Y_opt = opt_df[["velocity_x", "velocity_y", "velocity_z"]].values

    opt_scaler = StandardScaler()
    X_opt_scaled = opt_scaler.fit_transform(X_opt)

    split_opt = int(0.8 * len(X_opt_scaled))
    X_train_opt = X_opt_scaled[:split_opt]
    Y_train_opt = Y_opt[:split_opt]
    X_val_opt = X_opt_scaled[split_opt:]
    Y_val_opt = Y_opt[split_opt:]

    best_params = hyperparameter_optimization.optimize_hyperparameters(
        X_train_opt, Y_train_opt, X_val_opt, Y_val_opt, fs, serie, device, suffix="_incremental"
    )

    if args.base_model is not None and os.path.exists(args.base_model):
        try:
            import train_mlp
            h_layers, h_size = train_mlp.get_base_model_params(os.path.basename(args.base_model))
        except Exception:
            h_layers, h_size = best_params["hidden_layers"], best_params["hidden_size"]
        model = MLP(input_size, output_size, h_size, h_layers).to(device)
        model.load_state_dict(torch.load(args.base_model, map_location=device))
        scaler = joblib.load(args.base_model.replace(".pth", ".joblib"))
        print(f"Loaded base model and scaler from {args.base_model}")
    else:
        model = MLP(input_size, output_size, best_params["hidden_size"], best_params["hidden_layers"]).to(device)
        scaler = StandardScaler()

    results = []
    if SCATTERED:
        initial_blocks = [block.iloc[: len(block) // len(blocks)] for block in blocks]
        initial_df = pd.concat(initial_blocks).reset_index(drop=True)
        print(f"Initial training with {len(initial_df)} scattered samples")
        model = model_utils.train_on_block(
            model, scaler, initial_df, epochs=best_params["epochs"],
            device=device, lr=best_params["learning_rate"], batch_size=best_params["batch_size"]
        )

        initial_state = copy.deepcopy(model.state_dict())
        out_folder = os.path.join(config.MODEL_DIR, "incremental")
        os.makedirs(out_folder, exist_ok=True)

        for i, block in enumerate(blocks):
            is_holdout = args.holdout_last and (i == len(blocks) - 1)
            print(f"\n===== Processing block {i+1}/{len(blocks)} ({len(block)} samples) =====")
            
            block_model = MLP(input_size, output_size, best_params["hidden_size"], best_params["hidden_layers"]).to(device)
            block_model.load_state_dict(initial_state)
            
            if is_holdout:
                print("--> 🛡️ HOLDOUT MODE: Evaluating only. No fine-tuning on this block.")
            elif args.no_finetune:
                print("--> 🛑 [Test 1] NO FINE-TUNING MODE: Evaluating global scattered model only.")
            else:
                if args.subsequent_finetune_pct is not None:
                    start_idx = len(block) // len(blocks)
                    num_rows = int(len(block) * (args.subsequent_finetune_pct / 100.0))
                    end_idx = min(start_idx + num_rows, len(block))
                    finetune_block = block.iloc[start_idx:end_idx].reset_index(drop=True)
                    print(f"--> 📉 [Test 2] Fine-tuning on subsequent {args.subsequent_finetune_pct}% of data ({len(finetune_block)} samples).")
                else:
                    finetune_block = block
                    print("--> Fine-tuning on the entire block (Standard Scattered).")
                
                if len(finetune_block) > 0:
                    block_model = model_utils.train_on_block(
                        block_model, scaler, finetune_block, epochs=best_params["epochs_finetune"],
                        device=device, freeze=True, lr=best_params["learning_rate"], batch_size=best_params["batch_size"]
                    )

            metrics_dict = model_utils.evaluate_block(block_model, scaler, block, fs, device)
            metrics_dict["block"] = i
            metrics_dict["samples"] = len(block)
            metrics_dict["is_holdout"] = is_holdout
            results.append(metrics_dict)

            bloc_name = f"{serie}_block{i+1}"
            torch.save(block_model.state_dict(), os.path.join(out_folder, f"model_{bloc_name}.pth"))
            joblib.dump(scaler, os.path.join(out_folder, f"scaler_{bloc_name}.joblib"))
            
            if is_holdout:
                log_path = os.path.join(out_folder, f"blind_test_train_log_{serie}.txt")
                with open(log_path, "w") as f:
                    f.write(f"BLIND HOLDOUT RESULTS (Train Script) - Serie {serie}\nMode: SCATTERED\nRMSE on unseen Block {i+1}: {metrics_dict['rmse']:.6f}\n")
                print(f"--> Blind holdout text log saved to {log_path}")

    else:
        for i, block in enumerate(blocks):
            is_holdout = args.holdout_last and (i == len(blocks) - 1)
            print(f"\n===== Processing block {i+1}/{len(blocks)} ({len(block)} samples) =====")
            
            if is_holdout:
                print("--> 🛡️ HOLDOUT MODE: Evaluating only. No fine-tuning on this block.")
            else:
                if i == 0:
                    model = model_utils.train_on_block(
                        model, scaler, block, epochs=best_params["epochs"],
                        device=device, lr=best_params["learning_rate"], batch_size=best_params["batch_size"]
                    )
                else:
                    model = model_utils.train_on_block(
                        model, scaler, block, epochs=best_params["epochs_finetune"],
                        device=device, freeze=True, lr=best_params["learning_rate"], batch_size=best_params["batch_size"]
                    )
            
            metrics_dict = model_utils.evaluate_block(model, scaler, block, fs, device)
            metrics_dict["block"] = i
            metrics_dict["samples"] = len(block)
            metrics_dict["is_holdout"] = is_holdout
            results.append(metrics_dict)
            
            bloc_name = f"{serie}_block{i+1}"
            out_folder = os.path.join(config.MODEL_DIR, "incremental")
            os.makedirs(out_folder, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(out_folder, f"model_{bloc_name}.pth"))
            joblib.dump(scaler, os.path.join(out_folder, f"scaler_{bloc_name}.joblib"))

            if is_holdout:
                log_path = os.path.join(out_folder, f"blind_test_train_log_{serie}.txt")
                with open(log_path, "w") as f:
                    f.write(f"BLIND HOLDOUT RESULTS (Train Script) - Serie {serie}\nMode: SEQUENTIAL\nRMSE on unseen Block {i+1}: {metrics_dict['rmse']:.6f}\n")

    results_df = pd.DataFrame(results)
    res_path = os.path.join(config.DATA_DIR, "train", "results", f"results_{serie}", "block_metrics.csv")
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    results_df.to_csv(res_path, index=False)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if args.holdout_last:
        ax.plot(results_df["block"][:-1], results_df["rmse"][:-1], marker="o", label="Trained Blocks")
        ax.plot(results_df["block"].iloc[-1], results_df["rmse"].iloc[-1], marker="*", color="red", markersize=10, label="Blind Holdout")
        ax.legend()
    else:
        ax.plot(results_df["block"], results_df["rmse"], marker="o")
        
    ax.set_ylabel("RMSE")
    plt.tight_layout()
    plot_path = os.path.join(config.DATA_DIR, "train", "results", f"results_{serie}", "block_evolution.png")
    fig.savefig(plot_path)
    plt.close(fig)

    final_model_path = os.path.join(config.MODEL_DIR, "incremental", f"model_{serie}_final.pth")
    final_scaler_path = os.path.join(config.MODEL_DIR, "incremental", f"scaler_{serie}_final.joblib")
    torch.save(model.state_dict(), final_model_path)
    joblib.dump(scaler, final_scaler_path)

if __name__ == "__main__":
    main()
    