import argparse
import random

import torch
import torch.nn.functional as F

from dataset_skeleton import UCF101SkeletonDataset
from models.baseline_skeleton import BaselineSkeletonMLP
from models.skeleton_lstm import SkeletonLSTMModel


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_type, V, C, num_classes, weights_path, device):
    if model_type == "baseline":
        model = BaselineSkeletonMLP(
            num_keypoints=V,
            coord_dim=C,
            num_classes=num_classes,
            hidden_dim=256,
        )
    elif model_type == "lstm":
        model = SkeletonLSTMModel(
            num_keypoints=V,
            coord_dim=C,
            num_classes=num_classes,
            hidden_size=256,
            num_layers=2,
            dropout=0.3,
        )
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Predicción en UCF101 Skeleton subset")
    parser.add_argument(
        "--pkl_path",
        type=str,
        default="data/raw/ucf101_2d.pkl",
        help="Ruta al archivo .pkl de UCF101 skeleton",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test1",
        help="Nombre del split a usar (por defecto test1)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="lstm",
        choices=["baseline", "lstm"],
        help="Tipo de modelo a usar: baseline o lstm",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Ruta al archivo .pth con los pesos del modelo",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Índice de muestra en el dataset de test. Si no se da, se elige uno aleatorio.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Número de predicciones top-k a mostrar",
    )

    args = parser.parse_args()

    # Mismas 5 clases que en el entrenamiento
    subset_labels = [0, 1, 2, 3, 4]

    print(f"Cargando dataset {args.split}...")
    dataset = UCF101SkeletonDataset(
        pkl_path=args.pkl_path,
        split_name=args.split,
        num_frames=32,
        subset_labels=subset_labels,
        use_score=False,
    )

    # Sample de prueba para obtener shapes
    sample_skeleton, sample_label_idx = dataset[0]
    T, V, C = sample_skeleton.shape
    num_classes = len(dataset.label_to_idx)

    print(f"Dataset size: {len(dataset)} muestras")
    print(f"Shape de cada muestra: T={T}, V={V}, C={C}, num_classes={num_classes}")

    device = get_device()
    print("Usando device:", device)

    # Ruta por defecto de weights según el tipo de modelo
    if args.weights is None:
        if args.model_type == "baseline":
            weights_path = "baseline_skeleton_mlp.pth"
        else:
            weights_path = "skeleton_lstm_model.pth"
    else:
        weights_path = args.weights

    print(f"Cargando modelo '{args.model_type}' desde: {weights_path}")
    model = load_model(args.model_type, V, C, num_classes, weights_path, device)

    # Elegir índice
    if args.index is None:
        idx = random.randint(0, len(dataset) - 1)
        print(f"Ningún índice especificado, usando índice aleatorio: {idx}")
    else:
        idx = args.index
        if idx < 0 or idx >= len(dataset):
            raise ValueError(f"Índice fuera de rango: {idx}, tamaño dataset={len(dataset)}")

    skeleton, label_idx = dataset[idx]  
    label_original = dataset.idx_to_label[label_idx] 

    # Preparar batch de tamaño 1
    skeleton = skeleton.unsqueeze(0).to(device) 

    with torch.no_grad():
        logits = model(skeleton)         
        probs = F.softmax(logits, dim=1)  

    probs = probs.squeeze(0).cpu()       
    topk = min(args.topk, num_classes)
    top_probs, top_indices = torch.topk(probs, k=topk)

    print("\n=== Resultado de la predicción ===")
    print(f"Índice de muestra: {idx}")
    print(f"Etiqueta real (remapeada 0..{num_classes-1}): {label_idx}")
    print(f"Etiqueta original (ID en UCF101): {label_original}")

    print("\nTop-k predicciones:")
    for rank, (cls_idx, prob) in enumerate(zip(top_indices, top_probs), start=1):
        cls_idx = cls_idx.item()
        prob = prob.item()
        original_label = dataset.idx_to_label[cls_idx]
        print(f"  #{rank}: clase_idx={cls_idx} (label_original={original_label}) - prob={prob:.4f}")


if __name__ == "__main__":
    main()
