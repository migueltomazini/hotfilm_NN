<div style="display: flex; align-items: center;">
  <img src="TUCA_logo.jpg" alt="Turbulence Computing and Applications Group - ICMC - USP" width="100" height="100" style="margin-right: 20px;">
  <h1>Anemometric Data Conversion with MLP Neural Network</h1>
</div>

## Introduction

This project was developed as an undergraduate research project at the University of São Paulo, São Carlos campus, within the Turbulence Computing and Applications group at ICMC - USP. The research proposes an approach to convert voltage signals from hot-film sensors sampled at 2 kHz into three-dimensional velocity data using a Multi-Layer Perceptron (MLP) neural network, avoiding the need for King's Law or other analytical methods.

## Goal

The main goal is to provide a tool that converts hot-film voltage measurements into 3-component velocity data (x, y, z), assisted by low-frequency sonic anemometer measurements (20 Hz) used as reference. The MLP architecture was chosen for its simplicity and suitability for this problem.

## Motivation

King's Law, traditionally used to map hot-film voltage to velocity, can vary with environmental conditions (temperature, humidity, flow intensity), reducing conversion accuracy. An MLP offers a data-driven alternative that adapts to these variations and yields more robust estimations.

## Methodology

### Neural Network Structure

- **Inputs**: 3 voltages (axes x, y, z) and Reynolds number
- **Outputs**: 3 velocities (axes x, y, z)
- **Architecture**: Multi-Layer Perceptron (MLP) with configurable hidden layers and hidden units.

### Data Processing

1. **Data Collection**: Hot-film at 2 kHz and sonic anemometer at 20 Hz.
2. **Data Preparation**: Cleaning and aligning datasets for training and testing.
3. **Training**: Use of synthetic and real datasets to train the model with cross-validation to mitigate overfitting.
4. **Evaluation**: Metrics such as RMSE and spectral checks (periodograms) are used to verify that turbulence properties are preserved.

## Results

Tests with different datasets (clean, noisy, and datasets with sensor misalignment) demonstrate the MLP's ability to recover velocity from voltage.

- **Clean synthetic data**: Example RMSE ~ 0.065 m/s.
- **Noisy synthetic data**: Robust performance with errors < 4% in many cases.
- **Real data with angle misalignment**: Effective correction of geometric errors.

## Conclusions

The results indicate that data-driven neural models can provide reliable velocity estimates from hot-film sensors, simplifying data collection and improving robustness compared to traditional analytical approaches.

## Future Work

- Explore more advanced architectures (e.g., LSTM, temporal models) to capture time dependencies.
- Gather larger and more diverse real datasets to improve generalization.
- Extend the methodology to other sensing problems.

## Contributions

This repository provides an open-source implementation and documentation to support research and reproducibility in sensor data conversion.

---

Developed as part of an undergraduate research project at ICMC - USP, São Carlos.
