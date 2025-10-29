# GDG-Mamba: Dynamic Graph Embedding based on Mamba

## About the Project

DG-Mamba is a novel framework for dynamic graph embedding, designed to model complex, time-evolving networks efficiently. By leveraging the Mamba architecture, a state-space model with linear computational complexity, DG-Mamba addresses the scalability challenges associated with traditional transformer-based models. This project extends the capabilities of temporal graph representation learning through:

- Efficient handling of long-term dependencies in dynamic graphs.
- Integration of probabilistic embeddings for uncertainty quantification.
- Enhanced spatial and temporal representation learning.

This implementation is based on the research paper titled GDG-Mamba: Dynamic Graph Embedding based on Mamba.


## Features

- **Scalability:** Linear complexity for handling long graph sequences.
- **Probabilistic Embeddings:** Multivariate Gaussian distributions for each node representation.
- **Spatial and Temporal Integration:** Use of selective state-space mechanisms and Graph Isomorphism Network Edge (GINE) convolutions.
- **Applications:** Social network analysis, financial modeling, and biological system dynamics.

## How to Run the Repository

### Prerequisites

1. Python 3.8 or higher.
2. Required libraries:
    - `torch`
    - `torch-geometric`
    - `numpy`
    - `scipy`
    - `matplotlib`

Install the dependencies using:
```bash
pip install -r requirements.txt
```

### Downloading the Dataset

To download the required datasets, execute the `download_data.sh` script:

```bash
bash download_data.sh
```

The script performs the following steps:
1. Creates a `datasets` directory and downloads the required datasets.
2. Extracts and prepares the datasets for use in the experiments.
3. Includes popular datasets like SBM, Bitcoin, Slashdot, and others.

### Running an Example: Reality Mining Dataset

The Reality Mining dataset consists of human contact data among 100 students from MIT, collected over 9 months. Each node represents a student, and an edge denotes physical contact between two nodes.

#### Steps to Run:

1. **Prepare the Dataset:**
    - Ensure the dataset is downloaded and extracted using `download_data.sh`.

2. **Train the Model:**
    To train the model for DG-Mamba, navigate to the directory and simply run:
    ```bash
    python DG_Mamba.py
    ```

    For GDG-Mamba, execute:
    ```bash
    python GDG_Mamba.py
    ```

4. **Results:**
    The model generates node embeddings for each timestamp, which can be used for downstream tasks like link prediction and anomaly detection. Results include metrics such as Mean Average Precision (MAP) and Mean Reciprocal Rank (MRR).

### Key Configuration Options

The configurations, such as `lookback` and `embedding_size`, are set in the `config.json` file. You can modify these settings directly to suit your experiment needs.

## Citation

If you find this repository or the released model helpful, please cite our paper.



(If you have any questions or suggestions, please do not hesitate to contact us or open an issue on the GitHub repository. Your feedback is greatly appreciated!)

