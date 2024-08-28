# GNN Explanation Framework

This repository contains a framework for explaining and evaluating graph neural networks (GNNs). The primary focus is on assessing the causal effects of subgraphs in GNN predictions and mitigating the out-of-distribution (OOD) problem that arises when evaluating these explanations.

## Features

- **Graph Neural Network (GNN) Models**: Contains pre-trained GNN models for various datasets.
- **GNNExplainer**: Implements a meta GNN explainer to generate explanations for GNN predictions.
- **Evaluation Methods**: Introduces Deconfounded Subgraph Evaluation (DSE) to assess the fidelity of GNN explanations while addressing OOD problems.
- **Visualization**: Tools for visualizing subgraphs and their importance in GNN predictions.

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- PyTorch Geometric
- NetworkX
- Matplotlib
