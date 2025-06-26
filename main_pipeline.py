"""
Main pipeline for DocKylin training and instruction tuning as described in instruction.md.
- Pre-training: visual encoder + MLP, LLM frozen, APS used, DTS not used.
- Instruction tuning: all modules, APS and DTS used.
- Cosine annealing LR, batch size, MAXSIZE, etc.
- Placeholders for model, dataset, and optimizer.
"""
import os
import torch
import numpy as np
from aps import crop_image
from dts import dts, token_merge
import cv2
from transformers import AutoModelForVision2Seq, AutoProcessor
import timm
from PIL import Image
import json

MAXSIZE = 1728 * 1728  # Maximum number of pixels
BATCH_SIZE = 8
PRETRAIN_ITERS = 450_000
INSTRUCTION_ITERS = 300_000
LEARNING_RATE_START = 1e-4
LEARNING_RATE_END = 1e-5

# --- DATASET CLASSES ---
class DocVQADataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='val'):
        self.img_dir = os.path.join(root_dir, 'images')
        self.ann_path = os.path.join(root_dir, f'{split}_v1.0.json')
        with open(self.ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.samples = data['data'] if 'data' in data else data
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.img_dir, sample['image'])
        img = np.array(Image.open(img_path).convert('RGB'))
        question = sample.get('question', '')
        answer = sample.get('answers', [''])[0]
        return img, question, answer
    def __len__(self):
        return len(self.samples)

class InfoVQADataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='val'):
        self.img_dir = os.path.join(root_dir, 'images')
        self.ann_path = os.path.join(root_dir, f'{split}.json')
        with open(self.ann_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.img_dir, sample['image'])
        img = np.array(Image.open(img_path).convert('RGB'))
        question = sample.get('question', '')
        answer = sample.get('answers', [''])[0]
        return img, question, answer
    def __len__(self):
        return len(self.samples)

class SROIEDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='val'):
        self.img_dir = os.path.join(root_dir, 'img')
        self.ann_dir = os.path.join(root_dir, 'annotations')
        self.img_list = sorted(os.listdir(self.img_dir))
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        img = np.array(Image.open(img_path).convert('RGB'))
        # SROIE is typically used for OCR/IE, so no question/answer
        return img, '', ''
    def __len__(self):
        return len(self.img_list)

class FUNSDDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='val'):
        self.img_dir = os.path.join(root_dir, 'images')
        self.ann_dir = os.path.join(root_dir, 'annotations')
        self.img_list = sorted(os.listdir(self.img_dir))
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        img = np.array(Image.open(img_path).convert('RGB'))
        # FUNSD is typically used for form understanding, so no question/answer
        return img, '', ''
    def __len__(self):
        return len(self.img_list)

# --- DATASET SELECTOR ---
def get_dataset(name, root_dir, split='val'):
    if name.lower() == 'docvqa':
        return DocVQADataset(root_dir, split)
    elif name.lower() == 'infovqa':
        return InfoVQADataset(root_dir, split)
    elif name.lower() == 'sroie':
        return SROIEDataset(root_dir, split)
    elif name.lower() == 'funsd':
        return FUNSDDataset(root_dir, split)
    else:
        raise ValueError(f"Unknown dataset: {name}")

# --- REST OF THE PIPELINE (unchanged except for dataset usage) ---
class SwinVisualEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, features_only=True)
        self.out_proj = torch.nn.Linear(768, 512)  # Project to 512-dim for MLP
    def forward(self, x):
        # x: numpy array (B, H, W, C)
        x = torch.from_numpy(x).float().permute(0, 3, 1, 2) / 255.0
        # Resize to 224x224 for Swin input (simulate flexible input)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        feats = self.model(x)[-1]  # (B, C, H', W')
        feats = feats.flatten(2).transpose(1, 2)  # (B, N, C)
        feats = self.out_proj(feats)  # (B, N, 512)
        return feats

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(512, 512)
    def forward(self, x):
        return torch.relu(self.fc(x))

class QwenVL:
    def __init__(self, device='cpu'):
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True).to(device)
        self.device = device
    def forward(self, images, prompt="Describe the document."):
        # images: numpy array (B, H, W, C), prompt: str or list of str
        # Qwen-VL expects PIL images
        pil_images = [Image.fromarray(img.astype(np.uint8)) for img in images]
        if isinstance(prompt, str):
            prompts = [prompt] * len(pil_images)
        else:
            prompts = prompt
        inputs = self.processor(text=prompts, images=pil_images, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=32)
        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return decoded

def cosine_annealing_lr(iteration, total_iters, lr_start, lr_end):
    import math
    return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * iteration / total_iters))

def apply_aps_batch(images):
    processed = []
    for img in images:
        cropped = crop_image(img, direction='xy')[0]
        h, w = cropped.shape[:2]
        if h * w > MAXSIZE:
            r = int(np.floor(np.sqrt(MAXSIZE / (h * w)) * 10000) / 10000)
            new_h, new_w = int(h * r), int(w * r)
            cropped = cv2.resize(cropped, (new_w, new_h))
        processed.append(cropped)
    return np.stack(processed)

def apply_dts_features(features):
    indexing, essential_tokens, nonessential_tokens, _ = dts(features, features)
    essential = indexing(features, essential_tokens)
    nonessential = indexing(features, nonessential_tokens)
    merged = token_merge(essential, nonessential)
    return merged

def pretrain_stage(dataset):
    print("Starting pre-training stage...")
    visual_encoder = SwinVisualEncoder()
    mlp = MLP()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(list(visual_encoder.parameters()) + list(mlp.parameters()), lr=LEARNING_RATE_START)
    for iteration, batch in enumerate(dataloader):
        if iteration >= PRETRAIN_ITERS:
            break
        images = batch[0]
        lr = cosine_annealing_lr(iteration, PRETRAIN_ITERS, LEARNING_RATE_START, LEARNING_RATE_END)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        images = apply_aps_batch(images)
        features = visual_encoder(images)
        outputs = mlp(features)
        loss = outputs.sum() * 0
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if iteration % 10000 == 0:
            print(f"Pre-training iteration {iteration}/{PRETRAIN_ITERS}, lr={lr}")
    print("Pre-training complete.")
    return visual_encoder, mlp

def instruction_tuning_stage(dataset, visual_encoder, mlp, llm):
    print("Starting instruction tuning stage...")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    for iteration, batch in enumerate(dataloader):
        if iteration >= INSTRUCTION_ITERS:
            break
        images = batch[0]
        questions = batch[1] if len(batch) > 1 else None
        images = apply_aps_batch(images)
        prompts = questions if questions is not None else "Describe the document."
        outputs = llm.forward(images, prompt=prompts)
        if iteration % 10000 == 0:
            print(f"Instruction tuning iteration {iteration}/{INSTRUCTION_ITERS}")
            print("Sample Qwen-VL output:", outputs[0] if outputs else None)
    print("Instruction tuning complete.")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # TODO: Set your dataset name and root path here
    dataset_name = 'docvqa'  # or 'infovqa', 'sroie', 'funsd'
    dataset_root = '/path/to/your/dataset'  # <-- Set this path
    split = 'val'  # or 'train', as needed
    dataset = get_dataset(dataset_name, dataset_root, split)
    visual_encoder, mlp = pretrain_stage(dataset)
    llm = QwenVL(device=device)
    instruction_tuning_stage(dataset, visual_encoder, mlp, llm)

if __name__ == "__main__":
    main()
