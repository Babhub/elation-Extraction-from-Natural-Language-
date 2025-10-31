"""
Main script to train BoWClassifier on train/val datasets.
"""

import argparse
import joblib
import torch
from torch.utils.data import TensorDataset, DataLoader
from preprocess import get_data
from model import BoWClassifier, predict, get_best_model
from train import train_model
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train BoWClassifier')
    parser.add_argument('--train_path', default='./data/train.csv')
    parser.add_argument('--val_path', default='./data/val.csv')
    parser.add_argument('--labels_path', default='./data/all_labels.csv')
    parser.add_argument('--vectorizer_path', default='./vectorizer.joblib')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=5)
    return parser.parse_args()

def prepare_data_loaders(args, vectorizer, label_to_id):
    x_train, y_train = get_data(args.train_path, vectorizer, include_y=True, label_to_id=label_to_id)
    x_val, y_val = get_data(args.val_path, vectorizer, include_y=True, label_to_id=label_to_id)
    
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader

def main():
    args = parse_arguments()

    # Load labels
    with open(args.labels_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f]
    label_to_id = {label: i for i, label in enumerate(labels)}

    # Load vectorizer
    vectorizer = joblib.load(args.vectorizer_path)

    # Prepare data loaders
    train_loader, val_loader = prepare_data_loaders(args, vectorizer, label_to_id)

    # Initialize model
    input_size = len(vectorizer.vocabulary_)
    num_labels = len(label_to_id)
    model = get_best_model(input_size, num_labels)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train model
    trained_model, train_losses, val_losses = train_model(
        model, predict, train_loader, val_loader, device,
        lr=args.lr, num_epochs=args.num_epochs, patience=args.patience, save_path='best_model.pt'
    )

    # Plot loss curves
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()

if __name__ == "__main__":
    main()
