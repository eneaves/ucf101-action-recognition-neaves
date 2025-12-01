import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class UCF101SkeletonDataset(Dataset):
    def __init__(
        self,
        pkl_path: str,
        split_name: str,
        num_frames: int = 32,
        subset_labels: Optional[List[int]] = None,
        use_score: bool = False,
    ):
        super().__init__()

        self.pkl_path = pkl_path
        self.split_name = split_name
        self.num_frames = num_frames
        self.use_score = use_score

        # Cargar anotaciones
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        self.split = data["split"]
        self.annotations = data["annotations"]

        if split_name not in self.split:
            raise ValueError(
                f"Split '{split_name}' no encontrado en el .pkl. "
                f"Splits disponibles: {list(self.split.keys())}"
            )

        frame_dirs_in_split = set(self.split[split_name])

        # Crear dict 
        ann_dict = {ann["frame_dir"]: ann for ann in self.annotations}

        # Filtrar anotaciones que pertenecen al split
        samples = []
        for frame_dir in frame_dirs_in_split:
            ann = ann_dict.get(frame_dir, None)
            if ann is None:
                continue

            label = ann["label"]
            if subset_labels is not None and label not in subset_labels:
                continue

            samples.append(ann)

        if len(samples) == 0:
            raise ValueError(
                f"No se encontraron muestras para el split '{split_name}' "
                f"con subset_labels={subset_labels}."
            )

        # Guardamos las muestras
        self.samples = samples

        # Construimos mapping de labels usados
        unique_labels = sorted(list({ann["label"] for ann in self.samples}))
        self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}

        print(f"[UCF101SkeletonDataset] pkl: {pkl_path}")
        print(f"[UCF101SkeletonDataset] Split: {split_name}")
        print(f"[UCF101SkeletonDataset] Muestras: {len(self.samples)}")
        print(f"[UCF101SkeletonDataset] Labels únicos: {unique_labels}")
        print(f"[UCF101SkeletonDataset] Num clases: {len(unique_labels)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        ann = self.samples[idx]

        # keypoint
        keypoint = ann["keypoint"]      
        keypoint_score = ann["keypoint_score"]  
        label_original = ann["label"]

        # Usamos solo la primera persona M=0
        kp_person = keypoint[0]          
        score_person = keypoint_score[0]  

        # Número de frames reales
        T_total = kp_person.shape[0]

        # Obtener índices de frames muestreados
        frame_indices = self._sample_frame_indices(T_total, self.num_frames)

        kp_sampled = kp_person[frame_indices]        
        score_sampled = score_person[frame_indices] 

        # Normalización de coordenadas
        kp_sampled[..., 0] = (kp_sampled[..., 0] / 340.0) * 2 - 1
        kp_sampled[..., 1] = (kp_sampled[..., 1] / 256.0) * 2 - 1

        if self.use_score:
            score_expanded = score_sampled[..., None]
            kp_sampled = np.concatenate([kp_sampled, score_expanded], axis=-1) 

        # Convertir a tensor float32
        skeleton = torch.from_numpy(kp_sampled).float()  

        # Mapear label original al índice 0
        label_idx = self.label_to_idx[label_original]

        return skeleton, label_idx

    def _sample_frame_indices(self, num_total_frames: int, num_samples: int) -> List[int]:
        """
        Muestreo uniforme de `num_samples` frames de [0, num_total_frames-1].
        Si el video es corto, se repiten índices.
        """
        if num_total_frames < num_samples:
            indices = np.linspace(0, num_total_frames - 1, num_total_frames).astype(int)
            indices = np.pad(
                indices,
                (0, num_samples - num_total_frames),
                mode="edge"
            )
        else:
            indices = np.linspace(0, num_total_frames - 1, num_samples).astype(int)

        return indices.tolist()
