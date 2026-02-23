
 # User Manual

 ## Before You Start

 - After installing the Python virtual environment (see `installation.md`), change directory to the project root: `./hotfilm_NN`.
 - Activate the virtual environment:

 ```bash
 source hotfilm_env/bin/activate
 ```

 To deactivate:

 ```bash
 deactivate
 ```

 All commands assume the current working directory is `./hotfilm_NN`.

 Use a consistent series identifier for folders and files (e.g., `xxx`).

 ## Training

 ### Train a Network

 Requirements:

 - Hot-film voltage CSV sampled at 2000 Hz.
 - Sonic anemometer velocity CSV sampled at 20 Hz.

 1. Create a folder named `collected_data_xxx` under:

 ```
 data/raw_data/raw_train
 ```

 2. Place `hotfilm_xxx.csv` and `sonic_xxx.csv` in that folder.

 3. Generate the training CSV:

 ```bash
 python3 create_csv.py train {serie}
 ```

 4. Start training:

 ```bash
 python3 train_mlp.py {serie}
 ```

 Training outputs (models and metadata) are stored under `data/train/train_results/` and `models/`.

 ## Inference (Run)

 1. Create `collected_data_xxx` under `data/raw_data/raw_run` and place `hotfilm_xxx.csv` there.
 2. Prepare the run CSV:

 ```bash
 python3 create_csv.py run {serie}
 ```

 3. Run inference:

 ```bash
 python3 run_mlp.py {serie} {model_filename}.pth
 ```

 Outputs are saved under `data/run/results/velocity_{SERIE}/`.

 ## Spectrum and Dissipation Analysis (ε)

 Use `spectrum.py` to plot power spectral densities for predicted and sonic signals and optionally compute the dissipation rate ε.

 ### Required configuration for ε calculation

 Provide a JSON at `data/config/config_{SERIE}.json` with kinematic viscosity, sampling rates, `EPSILON_EXPECTED`, and optionally a theoretical spectrum file for validation.

 ### Run spectrum analysis

 ```bash
 python3 spectrum.py {SERIE}
 ```

 The script will prompt whether to compute ε and whether to run theoretical validation; both options require the config JSON.