import os
import torch
import numpy as np
from aps import crop_image
from dts import dts, token_merge
import cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import timm
from PIL import Image
from tqdm import tqdm
from main_pipeline import FUNSDDataset, SwinVisualEncoder, MLP, QwenVL, apply_aps_batch

def evaluate():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_root = 'FUNSD/dataset'
    split = 'val'
    dataset = FUNSDDataset(dataset_root, split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Load models (assumes you have saved checkpoints, or use fresh models for demo)
    visual_encoder = SwinVisualEncoder().to(device)
    mlp = MLP().to(device)
    llm = QwenVL(device=device)
    # Load fine-tuned Qwen model if available
    if os.path.exists("qwen_finetuned"):
        from transformers import Qwen2_5_VLForConditionalGeneration
        llm.model = Qwen2_5_VLForConditionalGeneration.from_pretrained("qwen_finetuned")
        llm.model.to(device)  # type: ignore
    # Load projection layer for text features
    hidden_dim = llm.model.config.hidden_size
    text_proj = torch.nn.Linear(hidden_dim, 512).to(device)
    # Load weights
    visual_encoder.load_state_dict(torch.load("visual_encoder.pth", map_location=device))
    mlp.load_state_dict(torch.load("mlp.pth", map_location=device))
    text_proj.load_state_dict(torch.load("text_proj.pth", map_location=device))
    visual_encoder.eval()
    mlp.eval()
    text_proj.eval()
    # llm.model.eval()  # QwenVL is already in eval mode by default

    num_epochs = 1  # You can increase this if needed
    for epoch in range(num_epochs):
        print(f"Starting evaluation on validation set, epoch {epoch+1}/{num_epochs}...")
        num_samples = 0
        num_exact_match = 0
        total_chars = 0
        correct_chars = 0
        for idx, batch in enumerate(tqdm(dataloader)):
            images = batch[0].numpy()
            gt_text = batch[1][0]  # ground truth text
            # Print the image filename
            img_name = dataset.img_list[idx]
            print(f"Evaluating image: {img_name}")
            images = apply_aps_batch(images)
            images = torch.as_tensor(images, dtype=torch.float32, device=device)
            features = visual_encoder(images.cpu().numpy())
            outputs = mlp(features)
            # For evaluation, use the LLM to generate output
            result = llm.forward(images.cpu().numpy(), prompt="what is the text in the image?")
            pred = result[0].strip()
            gt = gt_text.strip()
            print(f"Sample {idx}: Predicted: {pred} | Ground Truth: {gt}")
            # Metrics
            if pred == gt:
                num_exact_match += 1
            total_chars += len(gt)
            correct_chars += sum(p == g for p, g in zip(pred, gt))
            num_samples += 1
            # Optionally, break after a few samples for demo
            if idx >= 10:
                break
        print(f"Epoch {epoch+1}: Exact Match: {num_exact_match/num_samples:.2%}")
        print(f"Epoch {epoch+1}: Char Accuracy: {correct_chars/total_chars if total_chars > 0 else 0:.2%}")

if __name__ == "__main__":
    evaluate() 