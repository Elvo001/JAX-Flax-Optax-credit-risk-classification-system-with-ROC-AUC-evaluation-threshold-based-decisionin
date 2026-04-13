# JAX / Flax / Optax Credit Risk Classifier

A credit risk classification project built with JAX, Flax, and Optax, demonstrating how tabular banking-style prediction workflows can be implemented using a functional training loop.

## Overview

This project simulates a credit default prediction setting:

- Input: tabular applicant-style features
- Output: probability of default
- Decision: thresholded approve/reject-style binary classification

It is designed to showcase the migration from higher-level training workflows to explicit JAX-based model training.

## What this project demonstrates

- Flax model definition for tabular data
- Optax optimizer integration
- JIT-compiled training step
- Binary classification with sigmoid cross-entropy
- ROC-AUC and accuracy evaluation
- Threshold-based decisioning logic

## Project structure

```text
jax-flax-optax-credit-risk/
├── train_credit_risk_jax.py
├── requirements.txt
└── README.md
