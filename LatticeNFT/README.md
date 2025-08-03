

Lattice Neural Field Theory (LNFT)
A Physics-Inspired Architecture for Generative AI

This repository contains the official implementation of the Lattice Neural Field Theory (LNFT) model, a novel deep learning architecture inspired by principles from Lattice Quantum Chromodynamics (LQCD). This project uses a state-of-the-art, diffusion-style training and inference framework.

The core of this work is a Score-Based Model. Instead of learning an energy function, the model is trained to predict the noise in a corrupted data sample. This is achieved through a simple and stable Denoising Score Matching (DSM) objective. Generation is then performed using a fast, advanced ODE solver, DPM-Solver++, to transform pure noise into coherent data.

.
├── main.py             # Main script to run training and generation
├── training.py         # Contains the core training loop (train_ebm_model)
├── LNFT.py             # Main LNFT_EBM model class definition
├── LNFT_block.py       # The core GaugeConvolutionBlock and attention modules
├── base_modules.py     # Helper modules like PositionalEncoding
├── data_utils.py       # data manipulation/setup functions
├── tasks.py            # Inference/testing functions like manual_test_chat
├── sampler.py          # MALA Sampler method
├── requirements.txt    # Python pip librairies used in this project
└── README.md           # This file