"""Use incrementally trained block models to predict velocities on new/unseen data.

This script takes the sequence of models and scalers produced by incremental_train.py
and applies them sequentially to corresponding blocks of new data.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch

from utils import config, metrics, spectral_utils, validation_metrics, model_utils, validation_plots

device = torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(
        description="Apply incremental block models to new data",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("serie", help="series identifier (e.g. 0610)")
    parser.add_argument("--num-blocks", type=int, required=True, help="number of blocks to split the data into")
    parser.add_argument("--input", type=str, default=None, help="input CSV file")
    parser.add_argument("--output", type=str, default=None, help="output CSV file")
    parser.add_argument("--calc-metrics", action="store_true", help="if input has velocity columns, also compute RMSE/metrics")
    parser.add_argument("--holdout-last", action="store_true", help="Ensures the last block is evaluated as a completely blind test using the (N-1) model.")
    
    args = parser.parse_args()
    serie = args.serie

    input_file = args.input if args.input else os.path.join(config.DATA_DIR, "run", f"run_{serie}.csv")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    if "reynolds" not in df.columns:
        cfg_path = os.path.join(config.DATA_DIR, "config", f"config_{serie}.json")
        re_val = 0.0
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as fh:
                    cfg = json.load(fh)
                    re_val = cfg.get("RE_NUMBER", 0.0)
            except Exception:
                pass
        df["reynolds"] = re_val
        print(f"[Info] added missing reynolds column = {re_val}")

    indices = np.arange(len(df))
    split_indices = np.array_split(indices, args.num_blocks)
    all_preds = []

    print(f"Generating predictions using {args.num_blocks} incremental models...")
    for i, idx_list in enumerate(split_indices):
        block_num = i + 1
        block_df = df.iloc[idx_list].reset_index(drop=True)

        try:
            is_holdout = args.holdout_last and (i == len(split_indices) - 1)
            
            if is_holdout:
                print(f"--> 🛡️ [Holdout Test] Predicting block {block_num} using model strictly from block {block_num - 1}")
                model, scaler = model_utils.load_block_model_and_scaler(serie, block_num - 1, device)
            else:
                model, scaler = model_utils.load_block_model_and_scaler(serie, block_num, device)

            preds = model_utils.predict_on_block(model, scaler, block_df, device)
            all_preds.append(preds)
        finally:
            if 'model' in locals(): del model
            if 'scaler' in locals(): del scaler
            del block_df
            model_utils.cleanup_memory()

    final_preds = np.vstack(all_preds)
    df["velocity_predicted_x"] = final_preds[:, 0]
    df["velocity_predicted_y"] = final_preds[:, 1]
    df["velocity_predicted_z"] = final_preds[:, 2]

    if args.output is None:
        output_dir = os.path.join(config.DATA_DIR, "run", "results", f"velocity_{serie}")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"velocity_{serie}.csv")
    else:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        output_file = args.output

    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    del final_preds
    model_utils.cleanup_memory()

    # --- METRICS CALCULATION ---
    if args.calc_metrics:
        target_cols = ["velocity_x", "velocity_y", "velocity_z"]

        if all(col in df.columns for col in target_cols):
            print(f"\n{'='*50}")
            print(f"📊 VALIDATION: ARTICLE METRICS")
            print(f"{'='*50}")

            df_clean = df.dropna(subset=target_cols + ["velocity_predicted_x"])
            Y_true = df_clean[target_cols].values
            Y_pred = df_clean[["velocity_predicted_x", "velocity_predicted_y", "velocity_predicted_z"]].values

            with open(os.path.join(config.DATA_DIR, "config", f"config_{serie}.json")) as fh:
                cfg = json.load(fh)
            fs = cfg["FS_HOTFILM"]

            rmse_global = metrics.calculate_rmse(Y_pred, Y_true)
            deltas = metrics.calculate_delta_metrics(Y_true, Y_pred, fs)
            delta_general = np.mean(deltas)

            sk_preds_list = []
            sk_trues_list = []

            for idx_list in split_indices:
                block_clean = df.iloc[idx_list].dropna(subset=target_cols + ["velocity_predicted_x"])
                if len(block_clean) > 10: 
                    b_true = block_clean[target_cols].values
                    b_pred = block_clean[["velocity_predicted_x", "velocity_predicted_y", "velocity_predicted_z"]].values
                    
                    sk_preds_list.append(validation_metrics.calculate_velocity_derivative_skewness(b_pred, fs))
                    sk_trues_list.append(validation_metrics.calculate_velocity_derivative_skewness(b_true, fs))

            skewness_pred = {
                key: np.nanmean([d[key] for d in sk_preds_list]) for key in sk_preds_list[0]
            } if sk_preds_list else {'u_longitudinal': 0, 'u_lateral': 0, 'u_vertical': 0}

            skewness_true = {
                key: np.nanmean([d[key] for d in sk_trues_list]) for key in sk_trues_list[0]
            } if sk_trues_list else {'u_longitudinal': 0, 'u_lateral': 0, 'u_vertical': 0}

            output_text = f"{'='*50}\n"
            output_text += f"📊 VALIDATION: ARTICLE METRICS (Global Dataset)\n"
            output_text += f"{'='*50}\n\n"
            output_text += f"Records evaluated: {len(df_clean)}\n"
            output_text += f"Global Raw RMSE:  {rmse_global:.6f}\n"
            output_text += f"{'-'*50}\n"
            output_text += f"Delta_u1 (X-axis):  {deltas[0]:.4f}\n"
            output_text += f"Delta_u2 (Y-axis):  {deltas[1]:.4f}\n"
            output_text += f"Delta_u3 (Z-axis):  {deltas[2]:.4f}\n"
            output_text += f"Delta General:      {delta_general:.4f}\n"
            output_text += f"{'-'*50}\n"
            output_text += f"Skewness S_k (Pred vs True) [Expected u1 ~ -0.3]:\n"
            output_text += f"u1 (Longitudinal):  Pred={skewness_pred['u_longitudinal']:7.4f} | True={skewness_true['u_longitudinal']:7.4f}\n"
            output_text += f"u2 (Lateral):       Pred={skewness_pred['u_lateral']:7.4f} | True={skewness_true['u_lateral']:7.4f}\n"
            output_text += f"u3 (Vertical):      Pred={skewness_pred['u_vertical']:7.4f} | True={skewness_true['u_vertical']:7.4f}\n"
            output_text += f"{'-'*50}\n"

            print(output_text)
            metrics_file = os.path.join(output_dir, f"delta_metrics_{serie}.txt")
            with open(metrics_file, "w") as f:
                f.write(output_text)
            print(f"Global Metrics saved to {metrics_file}")
            
            if args.holdout_last:
                print(f"\n{'='*50}")
                print(f"🛡️ VALIDATION: BLIND HOLDOUT BLOCK ONLY")
                print(f"{'='*50}")
                
                last_block_indices = split_indices[-1]
                df_holdout = df.iloc[last_block_indices].dropna(subset=target_cols + ["velocity_predicted_x"])
                
                if len(df_holdout) > 0:
                    Y_true_h = df_holdout[target_cols].values
                    Y_pred_h = df_holdout[["velocity_predicted_x", "velocity_predicted_y", "velocity_predicted_z"]].values
                    
                    rmse_h = metrics.calculate_rmse(Y_pred_h, Y_true_h)
                    deltas_h = metrics.calculate_delta_metrics(Y_true_h, Y_pred_h, fs)
                    delta_general_h = np.mean(deltas_h)
                    skewness_pred_h = validation_metrics.calculate_velocity_derivative_skewness(Y_pred_h, fs)
                    skewness_true_h = validation_metrics.calculate_velocity_derivative_skewness(Y_true_h, fs)
                    
                    holdout_text = f"{'='*50}\n"
                    holdout_text += f"🛡️ BLIND HOLDOUT METRICS (Untainted Data)\n"
                    holdout_text += f"{'='*50}\n\n"
                    holdout_text += f"Records evaluated: {len(df_holdout)} (Block {args.num_blocks})\n"
                    holdout_text += f"Holdout Raw RMSE:  {rmse_h:.6f}\n"
                    holdout_text += f"{'-'*50}\n"
                    holdout_text += f"Delta_u1 (X-axis):  {deltas_h[0]:.4f}\n"
                    holdout_text += f"Delta_u2 (Y-axis):  {deltas_h[1]:.4f}\n"
                    holdout_text += f"Delta_u3 (Z-axis):  {deltas_h[2]:.4f}\n"
                    holdout_text += f"Delta General:      {delta_general_h:.4f}\n"
                    holdout_text += f"{'-'*50}\n"
                    holdout_text += f"Skewness S_k (Pred vs True):\n"
                    holdout_text += f"u1 (Longitudinal):  Pred={skewness_pred_h['u_longitudinal']:7.4f} | True={skewness_true_h['u_longitudinal']:7.4f}\n"
                    holdout_text += f"u2 (Lateral):       Pred={skewness_pred_h['u_lateral']:7.4f} | True={skewness_true_h['u_lateral']:7.4f}\n"
                    holdout_text += f"u3 (Vertical):      Pred={skewness_pred_h['u_vertical']:7.4f} | True={skewness_true_h['u_vertical']:7.4f}\n"
                    holdout_text += f"{'-'*50}\n"
                    
                    print(holdout_text)
                    blind_metrics_file = os.path.join(output_dir, f"blind_delta_metrics_{serie}.txt")
                    with open(blind_metrics_file, "w") as f:
                        f.write(holdout_text)
                    print(f"Blind Holdout Metrics saved to {blind_metrics_file}")
                
            del df_clean, Y_true, Y_pred
            model_utils.cleanup_memory()
        else:
            warning_msg = "\n[Warning] Velocity columns not found for metrics calculation."
            print(warning_msg)
            metrics_file = os.path.join(output_dir, f"delta_metrics_{serie}.txt")
            with open(metrics_file, "w") as f:
                f.write(warning_msg)

    print("\nIncremental block prediction complete.")

    if args.calc_metrics:
        try:
            validation_plots.generate_validation_plots(df, output_dir, serie, fs)
            validation_plots.generate_dissipation_series_plot(split_indices, df, output_dir, serie, fs)
        except Exception as e:
            print(f"[Proteção de Crash] Falha ao gerar gráficos de validação: {e}")

    # --- AUTOMATIC SPECTRAL ANALYSIS (MULTI-PLOT STYLE) ---
    print(f"\n{'='*50}")
    print("📊 STARTING SPECTRAL VALIDATION (PRED VS SONIC)")
    print(f"{'='*50}")

    sonic_file = os.path.join(config.DATA_DIR, "train", f"train_df_{serie}.csv")
    sonic_df = None
    if os.path.exists(sonic_file):
        print(f"[Spectral] Loading sonic reference from {sonic_file}")
        sonic_df = pd.read_csv(
            sonic_file,
            usecols=["time", "velocity_x", "velocity_y", "velocity_z"],
        )
        if len(sonic_df) > 500000:
            sonic_df = sonic_df.iloc[:500000].reset_index(drop=True)
    else:
        print(f"[Warning] Sonic file not found. Spectra will show predictions only.")

    spectral_dir = os.path.join(output_dir, "plots_spectral")
    os.makedirs(spectral_dir, exist_ok=True)

    with open(os.path.join(config.DATA_DIR, "config", f"config_{serie}.json")) as fh:
        cfg = json.load(fh)
    fs_hf = spectral_utils.estimate_sampling_frequency(df, "time")
    if fs_hf is None:
        fs_hf = cfg.get("FS_HOTFILM", 2000)

    fs_sonic = None
    if sonic_df is not None:
        fs_sonic = spectral_utils.estimate_sampling_frequency(sonic_df, "time")
    if fs_sonic is None:
        fs_sonic = cfg.get("FS_SONIC", 20.0)

    pred_cols = ["velocity_predicted_x", "velocity_predicted_y", "velocity_predicted_z"]

    print("Generating global spectral comparison...")
    try:
        spectral_utils.plot_combined_spectrum(
            df,  
            pred_cols,
            fs_hf,
            f"Global Spectral Analysis (Pred vs Sonic) - Serie {serie}",
            os.path.join(spectral_dir, f"combined_spectrum_global_{serie}.png"),
            sonic_df=sonic_df,
            fs_sonic=fs_sonic,
        )
    except Exception as e:
        print(f"[Proteção de Crash] Falha ao gerar espectro global: {e}")

    print(f"Generating spectral plots for {args.num_blocks} data blocks...")
    for i, idx_list in enumerate(split_indices):
        try:
            block_df_subset = df.iloc[idx_list][pred_cols]
            spectral_utils.plot_combined_spectrum(
                block_df_subset,
                pred_cols,
                fs_hf,
                f"Spectral Analysis Block {i+1} vs Sonic - Serie {serie}",
                os.path.join(spectral_dir, f"combined_spectrum_block_{i+1}_{serie}.png"),
                sonic_df=sonic_df,
                fs_sonic=fs_sonic,
            )
            print(f" -> Saved block {i+1}/{args.num_blocks}")
        except Exception as e:
            print(f"[Proteção de Crash] Falha ao gerar espectro do bloco {i+1}: {e}")
        finally:
            if 'block_df_subset' in locals(): del block_df_subset
            model_utils.cleanup_memory()

    print(f"\n[Done] All spectral plots are available in: {spectral_dir}")

if __name__ == "__main__":
    main()
    