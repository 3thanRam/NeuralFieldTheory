

Lattice Neural Field Theory (LNFT)
A Physics-Inspired Architecture for Generative AI

This repository contains the official implementation of the Lattice Neural Field Theory (LNFT) model, a novel deep learning architecture inspired by principles from Lattice Quantum Chromodynamics (LQCD). This project explores the use of gauge-covariant interactions and energy-based modeling as an alternative to standard Transformer architectures for generative tasks.

The core of this work is an Energy-Based Model (EBM) that learns a complex energy landscape over sequences of data. Generation is performed via Langevin Dynamics, treating text or other data sequences as states in a physical system to be evolved toward low-energy (high-coherence) configurations.

.
├── main.py             # Main script to run training and generation
├── training.py         # Contains the core training loop (train_ebm_model)
├── LNFT.py             # Main LNFT_EBM model class definition
├── LNFT_block.py       # The core GaugeConvolutionBlock and attention modules
├── base_modules.py     # Helper modules like PositionalEncoding
├── data_utils.py       # data manipulation/setup functions
├── tasks.py            # Inference/testing functions like manual_test_chat
└── README.md           # This file