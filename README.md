Temporal Convolutional Network (TCN) for Time-Series Forecasting

This project implements a Temporal Convolutional Network (TCN) — a deep learning architecture designed for sequence modeling and time-series analysis
While Convolutional Neural Networks (CNNs) were originally developed for image recognition, TCN adapts their core ideas to 1D temporal data, making it possible to efficiently capture temporal dependencies without relying on recurrent connections

Overview:
  TCN uses causal and dilated convolutions to model sequential data, allowing it to:
    Capture long-range dependencies through exponentially increasing receptive fields
    Benefit from parallel computation (unlike RNNs or LSTMs)
    Achieve stable training with fewer vanishing gradient issues

Architecture:
  The network consists of stacked dilated 1D convolutional layers with residual connections and normalization
  Each block processes the sequence over multiple time scales, enabling the model to learn both short-term and long-term temporal patterns

Features:
  Implementation based on PyTorch
  Supports customizable feature size, number of layers, and hidden dimensions
  Training and validation on CUDA device

Dataset:
  The dataset used in this project represents Bitcoin price dynamics over the last 2 years, providing a real-world financial time series for forecasting experiments
