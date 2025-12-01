import pickle
import os

def main():
    pkl_path = os.path.join("data", "raw", "ucf101_2d.pkl")  

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1") 

    print("Tipo de data:", type(data))
    print("Keys del dict:", data.keys())

    split = data["split"]
    print("\n=== SPLITS DISPONIBLES ===")
    for k, v in split.items():
        print(f"Split '{k}' tiene {len(v)} videos")

    annotations = data["annotations"]
    print("\nTotal de anotaciones:", len(annotations))

    ann0 = annotations[0]
    print("\n=== EJEMPLO DE ANOTACIÓN ===")
    print("Keys:", ann0.keys())
    print("frame_dir:", ann0["frame_dir"])
    print("label (int):", ann0["label"])
    print("total_frames:", ann0["total_frames"])
    print("img_shape:", ann0["img_shape"])
    print("original_shape:", ann0["original_shape"])
    print("keypoint shape:", ann0["keypoint"].shape)         
    print("keypoint_score shape:", ann0["keypoint_score"].shape)  
    labels = [ann["label"] for ann in annotations]
    print("\nLabel mínimo:", min(labels))
    print("Label máximo:", max(labels))
    print("Num labels distintos:", len(set(labels)))


if __name__ == "__main__":
    main()
