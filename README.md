# JE-IRT (Anonymous)

This is the anonymous GitHub repository for the paper:  
*A Geometric Lens on LLM Abilities through Joint Embedding Item Response Theory (JE-IRT)*.

## How to use

### 1. create a new conda environment called `je-irt`:

```bash
conda create -n je-irt python=3.10 -y
conda activate je-irt
```

Then, install the required packages from requirements.txt:
```bash
pip install -r requirements.txt
```
### 2. Download the Dataset
Download the **EmbedLLM** dataset from [Hugging Face](https://huggingface.co/datasets/RZ412/EmbedLLM) and place it under the `data/` directory.


### 3. Generate Question Embeddings

Since we use a frozen base encoder, we generate question embeddings beforehand for efficiency.

1. Open the notebook:
   ```bash
   jupyter notebook generate_embeddings.ipynb
   ```
2. Run all cells to generate embeddings and save them under embeddings/.



### 4. Train with Selected Hyperparameters

The training scripts with our selected hyperparameters are provided in:
- `run_modern-bert.sh`
- `run_sent-trans.sh`

1. Update Data Paths

    Edit each script to point to your local data files. Replace the arguments for:
    - `--train_csv`
    - `--val_csv`
    - `--test_csv`

2. Select Embedding Dimensions

    Each script contains command blocks for multiple embedding dimensions (e.g., 16, 64, 128, 256).
    Comment out the dimensions you do not want to run, and leave only the ones you wish to train.
    
3. Run Training

    Once configured, run the script for your chosen base encoder:
    ```bash
    bash run_modern-bert.sh
    # or
    bash run_sent-trans.sh
    ```

### 5. Adding Recent LLMs

Data for integrating recent LLMs is provided under the `new_LLM_data/` folder.  
