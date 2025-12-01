import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset_skeleton import UCF101SkeletonDataset
from models.baseline_skeleton import BaselineSkeletonMLP


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for skeletons, labels in dataloader:
        skeletons = skeletons.to(device)  
        labels = labels.to(device)       

        optimizer.zero_grad()
        logits = model(skeletons)         
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * skeletons.size(0)

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += skeletons.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for skeletons, labels in dataloader:
            skeletons = skeletons.to(device)
            labels = labels.to(device)

            logits = model(skeletons)
            loss = criterion(logits, labels)

            total_loss += loss.item() * skeletons.size(0)

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += skeletons.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def main():
    pkl_path = "data/raw/ucf101_2d.pkl"

    # Usamos split1 para train/val
    train_split = "train1"
    val_split = "test1"

    subset_labels = [0, 1, 2, 3, 4]

    print("Creando dataset de entrenamiento...")
    train_dataset = UCF101SkeletonDataset(
        pkl_path=pkl_path,
        split_name=train_split,
        num_frames=32,
        subset_labels=subset_labels,
        use_score=False,
    )

    print("\nCreando dataset de validación...")
    val_dataset = UCF101SkeletonDataset(
        pkl_path=pkl_path,
        split_name=val_split,
        num_frames=32,
        subset_labels=subset_labels,
        use_score=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # Info de shapes para configurar el modelo
    sample_skeletons, _ = next(iter(train_loader)) 
    _, T, V, C = sample_skeletons.shape
    num_classes = len(train_dataset.label_to_idx)

    print(f"\nShapes ejemplo: T={T}, V={V}, C={C}, num_classes={num_classes}")

    device = get_device()
    print("Usando device:", device)

    model = BaselineSkeletonMLP(
        num_keypoints=V,
        coord_dim=C,
        num_classes=num_classes,
        hidden_dim=256,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10  

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

    # Guardar el modelo entrenado
    torch.save(model.state_dict(), "baseline_skeleton_mlp.pth")
    print("\nModelo baseline guardado en baseline_skeleton_mlp.pth")


if __name__ == "__main__":
    main()
