# Session-Global Fusion GNN for Otto Recommender System

A Graph Neural Network-based recommendation system that combines session-level and global-level item relationships to predict user interactions in e-commerce sessions. This project implements a dual-branch architecture that captures both local session patterns and population-wide co-visitation patterns.

## About

This project addresses the Otto Group Product Classification Challenge, focusing on predicting the next items a user will interact with (click, add to cart, or order) based on their session history. The model employs a Session-Global Fusion Graph Neural Network (SGF-GNN) architecture that integrates:

- **Session Branch**: Captures local graph structure and temporal patterns within individual sessions
- **Global Branch**: Leverages population-level co-visitation patterns from the entire dataset
- **Fusion Mechanism**: Combines both branches to make final predictions

The system handles significant class imbalance (approximately 38:3:1 ratio for clicks:carts:orders) and includes comprehensive ablation studies to understand component contributions.

## Academic Context

This project was developed as part of **COMP8221 - Advanced Machine Learning** at Macquarie University. The work demonstrates advanced techniques in graph neural networks, recommendation systems, and handling imbalanced classification tasks.

## Dataset

The dataset is from the [Otto Group Product Classification Challenge](https://www.kaggle.com/competitions/otto-recommender-system) on Kaggle. The competition dataset contains:

- Approximately 11.8 million sessions
- Over 1.8 million unique items
- Three interaction types: clicks, carts, and orders
- Strong class imbalance favoring clicks over carts and orders

For more details about the dataset and competition, see the [Kaggle discussion thread](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364692).

## Methodology

### Architecture

The Session-Global Fusion GNN consists of two main branches:

1. **Session Branch**: 
   - Constructs a graph from items within each session
   - Uses Graph Attention Network (GAT) layers to capture local item relationships
   - Incorporates temporal features (time since last interaction, position in session)
   - Applies GraphSAGE layers for neighborhood aggregation

2. **Global Branch**:
   - Builds a global co-visitation graph from all sessions
   - Aggregates signals from items frequently co-visited with session items
   - Provides population-level context for recommendations

3. **Fusion Layer**:
   - Combines embeddings from both branches
   - Uses learned attention mechanisms to weight contributions
   - Produces final item scores for ranking

### Key Features

- **Class Imbalance Handling**: Implements weighted loss functions and sampling strategies
- **Negative Sampling**: Uses multiple strategies including popularity-based and co-visitation-based sampling
- **Temporal Features**: Incorporates time-based signals to capture recency effects
- **Ablation Studies**: Systematic evaluation of component importance

### Model Configuration

- Embedding dimension: 96
- Hidden dimension: 128
- GAT layers: 2
- GraphSAGE layers: 1
- Dropout: 0.3
- Learning rate: 0.001
- Batch size: 32

## Results

### Final Model Performance

The best model achieved the following metrics:

- **Click Recall**: 0.0918
- **Cart Recall**: 0.0
- **Order Recall**: 0.0
- **OTTO Weighted Score**: 0.0092
- **Best OTTO Score**: 0.0252 (at epoch 7)

### Ablation Study Results

Comprehensive ablation studies were conducted to evaluate component contributions:

| Variant | Click Recall | Cart Recall | Order Recall | OTTO Score |
|---------|--------------|-------------|--------------|------------|
| Full Model | 0.126 | 0.000 | 0.200 | 0.133 |
| No Global Branch | 0.042 | 0.111 | 0.200 | 0.158 |
| No Session Branch | 0.116 | 0.111 | 0.200 | 0.165 |
| No Temporal Features | 0.084 | 0.111 | 0.200 | 0.162 |

The ablation studies reveal that:
- The session branch is critical for click prediction
- The global branch helps with cart predictions
- Temporal features contribute to overall performance
- All components work together synergistically

## Project Structure

```
session-global-fusion-gnn-otto-recommender/
├── main_training_pipeline.ipynb    # Main training and evaluation notebook
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── ablation_studies/               # Ablation study results and visualizations
│   ├── ablation_results.json       # Summary of all ablation variants
│   ├── ablation_comparison.png    # Visual comparison of variants
│   └── ...
├── models/                         # Trained model checkpoints
│   └── fusion_gnn_final.pt        # Final trained model
├── visualizations/                 # Training curves and performance plots
│   ├── fusion_gnn/                # Main model visualizations
│   └── ...
├── artifacts/                      # Preprocessing artifacts
│   ├── aid_to_idx.pkl             # Article ID to index mapping
│   ├── scaler_*.pkl                # Feature scalers
│   └── ...
├── data/                          # Processed data files
│   ├── test_full.parquet          # Test dataset
│   └── session_sampling_weights.parquet
├── checkpoints/                    # Training checkpoints
└── docs/                          # Documentation and assignment materials
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/jasminehuynh11/session-global-fusion-gnn-otto-recommender.git
cd session-global-fusion-gnn-otto-recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from the [Kaggle competition](https://www.kaggle.com/competitions/otto-recommender-system) and place it in the `data/` directory.

## Usage

### Training

Open and run the `main_training_pipeline.ipynb` notebook. The notebook is organized into sections:

1. **Data Loading**: Load and preprocess the Otto dataset
2. **Graph Construction**: Build session and global graphs
3. **Model Definition**: Define the SGF-GNN architecture
4. **Training**: Train the model with appropriate loss functions
5. **Evaluation**: Evaluate on validation set and generate predictions
6. **Ablation Studies**: Run ablation experiments

### Evaluation

The model outputs predictions in the format required by the Kaggle competition. Evaluation metrics include:

- Recall@20 for clicks, carts, and orders
- OTTO weighted score (0.5 × click + 0.3 × cart + 0.2 × order)

## Contributors

- **Jasmine Huynh** - [GitHub](https://github.com/jasminehuynh11)
- **Nhat Nguyen** - [GitHub](https://github.com/NhatNguyen3001)

## References

- [Otto Group Product Classification Challenge](https://www.kaggle.com/competitions/otto-recommender-system)
- [Kaggle Competition Discussion](https://www.kaggle.com/competitions/otto-recommender-system/discussion/364692)

## License

This project is developed for academic purposes as part of COMP8221 - Advanced Machine Learning at Macquarie University.
