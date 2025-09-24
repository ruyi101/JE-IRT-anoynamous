import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

class IRTTrainer:
    def __init__(self, 
                 model, 
                 train_loader, 
                 val_loader, 
                 embedding_mapper, 
                 optimizer,
                 scheduler=None, 
                 test_loader=None, 
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.embedding_mapper = embedding_mapper
        self.criterion = nn.BCEWithLogitsLoss()
        self.scheduler = scheduler

    def _step(self, batch):
        prompt_ids = batch["prompt_id"]  # (B,)
        model_names = batch["model_name"]
        labels = batch["label"].float().to(self.device)  # (B,)

        embeddings = torch.tensor([self.embedding_mapper[str(id)] for id in prompt_ids]).to(self.device)  # (B, D)

        b, llm_emb = self.model(embeddings, model_names)  # (B, D) each

        # Compute the difference vector
        diff = llm_emb - b  # shape: (B, D)

        # Dot product: bᵀ (llm_emb - b)
        numerator = torch.sum(b * diff, dim=-1)  # shape: (B,)

        # L2 norm of b
        denominator = b.norm(p=2, dim=-1) + 1e-8  # shape: (B,)

        # Final scalar score per sample
        logits = numerator / denominator  # shape: (B,)

        # Apply sigmoid for probability
        probs = torch.sigmoid(logits)  # shape: (B,)

        loss = self.criterion(logits, labels)

        return loss, probs, labels

    def train_epoch(self):
        self.model.train()

        for batch in tqdm(self.train_loader, desc="Training"):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)

            self.optimizer.zero_grad()
            loss, probs, labels = self._step(batch)
            loss.backward()
            self.optimizer.step()

        correct = ((probs > 0.5) == labels.bool()).sum().item()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item(), correct / labels.size(0)

    def evaluate(self):
        def run_eval(loader, desc):
            self.model.eval()
            total_loss, total_correct, total = 0.0, 0, 0

            with torch.no_grad():
                for batch in tqdm(loader, desc=desc):
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(self.device)

                    loss, probs, labels = self._step(batch)
                    total_loss += loss.item() * labels.size(0)
                    total_correct += ((probs > 0.5) == labels.bool()).sum().item()
                    total += labels.size(0)

            return total_loss / total, total_correct / total

        val_loss, val_acc = run_eval(self.val_loader, "Evaluating (Val)")

        if self.test_loader is not None:
            test_loss, test_acc = run_eval(self.test_loader, "Evaluating (Test)")
        else:
            test_loss, test_acc = None, None

        return val_loss, val_acc, test_loss, test_acc
    

    def predict(self, loader):
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                _, probs, _ = self._step(batch)
                preds = (probs > 0.5).long()  # convert to binary class
                all_preds.extend(preds.cpu().tolist())

        return all_preds




