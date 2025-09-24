from torch.utils.data import Dataset
import torch

class IRTDataset(Dataset):
    def __init__(self, dataframe, return_metadata=False):
        """
        Args:
            dataframe: pandas DataFrame with required columns
            return_metadata: if True, also return prompt_id, model_id, category_id, category
        """
        self.df = dataframe.reset_index(drop=True)
        self.return_metadata = return_metadata

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = {
            'prompt_id': row['prompt_id'],
            'model_name': row['model_name'],
            'label': row['label']
        }
        if self.return_metadata:
            sample.update({
                'prompt': row['prompt'],
                'model_id': row['model_id'],
                'category_id': row['category_id'],
                'category': row['category']
            })
        return sample

def irt_collate_fn(batch):
    prompt_id = [item['prompt_id'] for item in batch]
    model_name = [item['model_name'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])

    result = {
        'prompt_id': prompt_id,
        'model_name': model_name,
        'label': labels
    }

    # Add metadata fields if present
    for field in ['prompt', 'model_id', 'category_id', 'category']:
        if field in batch[0]:
            result[f'{field}'] = [item[field] for item in batch]

    return result



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # 'min' for loss, 'max' for accuracy
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False

        improvement = (current_score < self.best_score - self.min_delta) if self.mode == 'min' \
                      else (current_score > self.best_score + self.min_delta)

        if improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

