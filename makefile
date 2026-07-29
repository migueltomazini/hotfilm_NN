# ==============================================================================
# Pipeline Execution Makefile for Hot-Film Neural Network
# ==============================================================================
# Usage:
#   make run                          Run full pipeline for default SERIE (0610)
#   make run SERIE=5940               Run full pipeline for series 5940
#   make clean SERIE=0610              Clean output results for a specific series
# ==============================================================================

# --- Default Parameters ---
SERIE          ?= 0610
NUM_BLOCKS     ?= 10
PYTHON         ?= python3

# --- Target Definitions ---
.PHONY: all train predict spectrum compare run clean help

# Default target when calling `make` without arguments
all: run

help:
	@echo "=============================================================================="
	@echo "  Hot-Film NN Pipeline Automation"
	@echo "=============================================================================="
	@echo "  Available Commands:"
	@echo "    make run SERIE=0610          Execute entire pipeline (train -> predict -> spectrum -> compare)"
	@echo "    make train SERIE=0610        Run incremental training with Sobolev loss"
	@echo "    make predict SERIE=0610      Run incremental prediction on trained blocks"
	@echo "    make spectrum SERIE=0610     Compute velocity spectra"
	@echo "    make compare SERIE=0610      Generate Figures 4, 5, 6, and 7 comparisons"
	@echo "    make clean SERIE=0610        Remove generated figures and CSV results for SERIE"
	@echo "=============================================================================="

# ------------------------------------------------------------------------------
# 1. Incremental Training (Scattered Mode + 10/10/80 Split + Sobolev Loss)
# ------------------------------------------------------------------------------
train:
	@echo "\n[Step 1/4] Starting Incremental Sobolev Training for Serie: $(SERIE)..."
	$(PYTHON) incremental_train.py $(SERIE) \
		--num-blocks $(NUM_BLOCKS) \
		--scattered \

# ------------------------------------------------------------------------------
# 2. Incremental Prediction
# ------------------------------------------------------------------------------
predict:
	@echo "\n[Step 2/4] Running Incremental Prediction for Serie: $(SERIE)..."
	$(PYTHON) incremental_predict.py $(SERIE) \
		--num-blocks $(NUM_BLOCKS) \
		--calc-metrics \
		--input data/train/train_df_$(SERIE).csv

# ------------------------------------------------------------------------------
# 3. Spectrum Calculation
# ------------------------------------------------------------------------------
spectrum:
	@echo "\n[Step 3/4] Computing Energy Spectra for Serie: $(SERIE)..."
	$(PYTHON) spectrum.py $(SERIE)

# ------------------------------------------------------------------------------
# 4. Article Comparison (Figures 4, 5, 6, and 7 Generation)
# ------------------------------------------------------------------------------
compare:
	@echo "\n[Step 4/4] Generating Article Comparison Plots (Fig 4, 5, 6, 7)..."
	$(PYTHON) article_comparison.py $(SERIE)

# ------------------------------------------------------------------------------
# Master Target: Executes Full Pipeline Sequentially
# ------------------------------------------------------------------------------
run: train predict spectrum compare
	@echo "\nFull pipeline completed successfully for Serie $(SERIE)!"

# ------------------------------------------------------------------------------
# Cleanup Target
# ------------------------------------------------------------------------------
clean:
	@echo "\nCleaning generated files for Serie: $(SERIE)..."
	rm -rf data/run/results/velocity_$(SERIE)/article_comparison/*.png
	rm -rf data/train/results/results_$(SERIE)/*.csv
	rm -rf data/train/results/results_$(SERIE)/*.png
	@echo "Clean finished."