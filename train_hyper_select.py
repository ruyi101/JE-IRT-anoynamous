import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
import os
import random
import numpy as np


from model import IRTembed
from utils import IRTDataset, irt_collate_fn, EarlyStopping
from trainer_scheduler import IRTTrainer





def main():
    """
    Main training function for the IRTembed model.

    This script loads a specified embedding model (e.g., Qwen), optionally applies LoRA tuning,
    reads training and validation data from CSV files, and trains a model that projects prompts 
    and language models into a shared latent space using an Item Response Theory-inspired formulation.

    Command-line arguments:
        --model_id:          Hugging Face model ID for the base encoder (default: Qwen3-Embedding-0.6B)
        --train_csv:         Path to training CSV file (must include columns like 'prompt', 'model_name', 'label')
        --val_csv:           Path to validation CSV file
        --embed_dim:         Dimensionality of the embedding space (default: 512)
        --batch_size:        Training batch size (default: 16)
        --lr:                Learning rate (default: 1e-4)
        --weight_decay:      Weight decay (default: 0.01)
        --epochs:            Number of training epochs (default: 10)
        --output_path:       File path to save the best model (default: ./best_model.pt)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder", type=str, default="qwen-3-0.6")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=False, default=None)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output_path", type=str, default="./")
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--gamma", type=float, default=0.9)

    args = parser.parse_args()



    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)


    if args.encoder == "sentence-transformer":
        encoder_embedding_dim = 768
        embedding_mapper = json.load(open("/embeddings/sentence-transformer.json", "r"))
    elif args.encoder == "modern-bert":
        encoder_embedding_dim = 1024
        embedding_mapper = json.load(open("/embeddings/modern-bert.json", "r"))
    else:
        raise ValueError(f"Unsupported encoder: {args.encoder}")

    # TensorBoard writer for logging
    writer = SummaryWriter(log_dir=f"runs/{args.encoder}_embed{args.embed_dim}_ba{args.batch_size}_lr{args.lr}_wd{args.weight_decay}_dropout{args.dropout}_gamma{args.gamma}")



    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv) if args.test_csv else None

    llm_names = set(train_df['model_name'].unique()).union(set(val_df['model_name'].unique()))
    llm_names = list(llm_names)
    llm_names = sorted(llm_names)  # Sort for consistent ordering

    train_loader = DataLoader(IRTDataset(train_df), batch_size=args.batch_size, shuffle=True, collate_fn=irt_collate_fn)
    val_loader = DataLoader(IRTDataset(val_df), batch_size=args.batch_size, shuffle=False, collate_fn=irt_collate_fn)
    test_loader = DataLoader(IRTDataset(test_df), batch_size=args.batch_size, shuffle=False, collate_fn=irt_collate_fn) if test_df is not None else None





    model = IRTembed(
        llm_names=llm_names,
        embedding_dim = encoder_embedding_dim,
        llm_embed_dim=args.embed_dim,
        dropout=args.dropout,
        device="cuda",
    )



    # Optimizer setup
    # optimizer = torch.optim.SGD(
    # filter(lambda p: p.requires_grad, model.parameters()),
    # lr=args.lr,
    # momentum=0.9,  # recommended
    # weight_decay=args.weight_decay
    # )

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    early_stopper = EarlyStopping(patience=10, min_delta=0.0, mode='min')  # for val_loss



    if args.scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=20)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=args.lr * 0.05)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")

    # Training loop
    trainer = IRTTrainer(model, optimizer=optimizer, scheduler=scheduler, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, 
                         embedding_mapper=embedding_mapper, device="cuda")
    best_val_loss = float("inf")
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = trainer.train_epoch()
        val_loss, val_acc, test_loss, test_acc = trainer.evaluate()

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        if test_loader is not None:
            writer.add_scalar("Loss/Test", test_loss, epoch)
            writer.add_scalar("Accuracy/Test", test_acc, epoch)
        
        if val_loss < best_val_loss:
            output_file = os.path.join(args.output_path, f"{args.encoder}/embed{args.embed_dim}_ba{args.batch_size}_lr{args.lr}_wd{args.weight_decay}_dropout{args.dropout}_gamma{args.gamma}_loss.pt")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            print("New best model found. Saving...")
            torch.save(model.state_dict(), output_file)
            best_val_loss = val_loss
            best_val_acc = val_acc

        if early_stopper(val_loss):
            print("Early stopping triggered.")
            break

        


    writer.close()

if __name__ == "__main__":
    main()
