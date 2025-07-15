import os, json
from tqdm import tqdm

dataset_dir = "FUNSD/dataset/training_data"
img_dir = os.path.join(dataset_dir, "images")
ann_dir = os.path.join(dataset_dir, "annotations")

output_path = "data/funsd_pretrain_manifest.jsonl"
os.makedirs("data", exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f_out:
    for filename in tqdm(sorted(os.listdir(img_dir))):
        if not filename.endswith(".png"):
            continue
        base = os.path.splitext(filename)[0]
        img_path = os.path.join(img_dir, filename)
        ann_path = os.path.join(ann_dir, base + ".json")
        if not os.path.exists(ann_path):
            continue
        with open(ann_path, encoding="utf-8") as f_ann:
            ann = json.load(f_ann)
        full_text = " ".join(
            field.get("text", "") for field in ann.get("form", []) if field.get("text", "")
        ).strip()
        if full_text:
            f_out.write(json.dumps({"image": img_path, "text": full_text}) + "\n")

print("✅ FUNSD manifest written to", output_path)
