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
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration
from transformers.tokenization_utils import PreTrainedTokenizer
import timm
from PIL import Image
import json
from Dataset.download import download_funsd
from tqdm import tqdm
from transformers.utils.quantization_config import BitsAndBytesConfig


MAXSIZE = 1728 * 1728  # Maximum number of pixels
BATCH_SIZE = 1
PRETRAIN_ITERS = 450_000
INSTRUCTION_ITERS = 300_000
LEARNING_RATE_START = 1e-4
LEARNING_RATE_END = 1e-5

# --- DATASET CLASSES ---
class FUNSDDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='val', image_size=(1000, 1000)):
        if split == 'train':
            data_folder = 'training_data'
        else:
            data_folder = 'testing_data'
        self.img_dir = os.path.join(root_dir, data_folder, 'images')
        self.ann_dir = os.path.join(root_dir, data_folder, 'annotations')
        self.img_list = sorted(os.listdir(self.img_dir))
        self.image_size = image_size
    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        ann_name = os.path.splitext(img_name)[0] + '.json'
        ann_path = os.path.join(self.ann_dir, ann_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.image_size)
        img = np.array(img)
        # Load annotation and extract ground truth text
        gt_text = ""
        if os.path.exists(ann_path):
            with open(ann_path, 'r', encoding='utf-8') as f:
                ann = json.load(f)
            gt_text = " ".join([field.get("text", "") for field in ann.get("form", []) if field.get("text", "")])
        return img, gt_text
    def __len__(self):
        return len(self.img_list)

# --- DATASET SELECTOR ---
def get_dataset(name, root_dir, split='val'):
    if name.lower() == 'funsd':
        return FUNSDDataset(root_dir, split)
    else:
        raise ValueError(f"Unknown dataset: {name}")

# --- REST OF THE PIPELINE (unchanged except for dataset usage) ---
class SwinVisualEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            features_only=True,         # gives list of feature maps
            out_indices=(3,)            # final stage is index 3
        )
        # print("Number of output stages available:", len(self.model.feature_info))  # Debug
        self.out_proj = torch.nn.Linear(768, 512)

    def forward(self, x):
        # numpy (B, H, W, C) → torch (B, C, H, W)
        x = torch.as_tensor(x, dtype=torch.float32, device=self.out_proj.weight.device)
        x = x.permute(0, 3, 1, 2) / 255.0
        x = torch.nn.functional.interpolate(x, size=(224, 224),
                                            mode='bilinear', align_corners=False)
        feats = self.model(x)[-1]
        if isinstance(feats, torch.Tensor):
            if len(feats.shape) == 4 and feats.shape[1] < feats.shape[-1]:  # Channels-last
                feats = feats.permute(0, 3, 1, 2)  # (B, C, H, W)
            feats = feats.flatten(2).transpose(1, 2).contiguous()  # (B, N, C)
            feats = self.out_proj(feats)  # (B, N, 512)
            return feats
        else:
            raise TypeError("Expected tensor from visual encoder, got: " + str(type(feats)))

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(512, 512)
    def forward(self, x):
        return torch.relu(self.fc(x))

class QwenVL:
    def __init__(self, device='cpu'):
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,  # or use load_in_4bit=True
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,
            llm_int8_enable_fp32_cpu_offload=True,  # offload overflow to CPU
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            quantization_config=bnb_config,
            device_map="auto"
        )


        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        self.device = device

        # Ensure a unique pad token
        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.pad_token = '[PAD]'
            self.model.resize_token_embeddings(len(tokenizer))

    def forward(self, images, prompt="Describe the document."):
        from PIL import Image
        pil_images = [Image.fromarray(img.astype(np.uint8)) for img in images]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt if isinstance(prompt, str) else prompt[0]},
                ],
            },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[pil_images[0]], return_tensors="pt").to(self.model.device)
        generate_ids = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.get("attention_mask", None),
            max_new_tokens=30
        )
        output = self.processor.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return [output]

def cosine_annealing_lr(iteration, total_iters, lr_start, lr_end):
    import math
    return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * iteration / total_iters))

def apply_aps_batch(images):
    processed = []
    target_size = (1000, 1000)  # (width, height)
    for img in images:
        # If img is a torch tensor, convert to numpy
        if hasattr(img, 'numpy'):
            img = img.numpy()
        if 'torch' in str(type(img)):
            img = img.detach().cpu().numpy()
        cropped = crop_image(img, direction='xy')[0]
        # Resize after cropping to ensure consistent shape
        cropped = cv2.resize(cropped, target_size)
        processed.append(cropped)
    return np.stack(processed)

def apply_dts_features(features):
    indexing, essential_tokens, nonessential_tokens, _ = dts(features, features)
    essential = indexing(features, essential_tokens)
    nonessential = indexing(features, nonessential_tokens)
    merged = token_merge(essential, nonessential)
    return merged

def contrastive_loss(image_features, text_features, temperature=0.07):
    import torch.nn.functional as F
    # Normalize
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    # Ensure same dtype and device
    text_features = text_features.to(dtype=image_features.dtype, device=image_features.device)
    logits = image_features @ text_features.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2

def pretrain_stage(dataset, num_epochs=1):
    print("Starting pre-training stage...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    visual_encoder = SwinVisualEncoder().to(device)
    mlp = MLP().to(device)
    llm = QwenVL(device=device)
    hidden_dim = llm.model.config.hidden_size  # 4096 for Qwen-2.5-VL
    # a single linear layer to match dims (moves to same device as visual encoder later)
    text_proj = torch.nn.Linear(hidden_dim, 512).to(device)
    for param in llm.model.parameters():
        param.requires_grad = False
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(
        list(visual_encoder.parameters())
        + list(mlp.parameters())
        + list(text_proj.parameters()),
        lr=LEARNING_RATE_START
    )
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        print(f"Epoch {epoch+1}/{num_epochs}")
        for iteration, batch in enumerate(tqdm(dataloader)):
            images = batch[0]
            annotation_texts = batch[1]
            # Print image filenames for this batch
            if hasattr(dataset, 'img_list'):
                batch_indices = range(iteration * BATCH_SIZE, min((iteration + 1) * BATCH_SIZE, len(dataset)))
                img_names = [dataset.img_list[i] for i in batch_indices if i < len(dataset)]
                print(f"Training batch {iteration}: {img_names}")
            lr = cosine_annealing_lr(iteration, PRETRAIN_ITERS, LEARNING_RATE_START, LEARNING_RATE_END)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            images = apply_aps_batch(images)
            images = torch.as_tensor(images, dtype=torch.float32, device=device)
            # Get image features
            features = visual_encoder(images)
            img_embeds = mlp(features).mean(dim=1)  # (B, 512)
            # Get text features from frozen LLM (batched)
            with torch.no_grad():
                tokens = llm.processor.tokenizer(list(annotation_texts), return_tensors="pt", padding=True, truncation=True, max_length=512).to(llm.model.device)
                outputs = llm.model.model.language_model(**tokens, output_hidden_states=True, return_dict=True)
                last_hidden = outputs.hidden_states[-1].mean(dim=1)  # (B, 4096)
            dtype = text_proj.weight.dtype
            text_embeds = text_proj(last_hidden.to(device=device, dtype=dtype))  # (B, 512)
            text_embeds = text_embeds.to(dtype=img_embeds.dtype, device=img_embeds.device)  # Ensure same dtype/device
            # Compute contrastive loss
            loss = contrastive_loss(img_embeds, text_embeds)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            num_batches += 1
            if iteration % 10000 == 0:
                print(f"Pre-training iteration {iteration}, lr={lr}, loss={loss.item():.4f}")
        print(f"Epoch {epoch+1} average loss: {epoch_loss/num_batches:.4f}")
    print("Pre-training complete.")
    # Save model weights
    torch.save(visual_encoder.state_dict(), "visual_encoder.pth")
    torch.save(mlp.state_dict(), "mlp.pth")
    torch.save(text_proj.state_dict(), "text_proj.pth")
    return visual_encoder, mlp

def instruction_tuning_stage(dataset, visual_encoder, mlp, llm, num_epochs=1):
    print("Starting instruction tuning stage...")
    # Unfreeze Qwen parameters for tuning
    for param in llm.model.parameters():
        param.requires_grad = True
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    # Include Qwen parameters in optimizer
    optimizer = torch.optim.Adam(
        list(visual_encoder.parameters()) + list(mlp.parameters()) + list(llm.model.parameters()),
        lr=LEARNING_RATE_START
    )
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        print(f"Instruction tuning epoch {epoch+1}/{num_epochs}")
        for iteration, batch in enumerate(tqdm(dataloader)):
            images = batch[0]
            annotation_texts = batch[1]
            images = apply_aps_batch(images)
            images = torch.as_tensor(images, dtype=torch.float32, device=llm.model.device)
            features = visual_encoder(images)
            outputs = mlp(features)
            print("-"*40)
            pil_images = [Image.fromarray(img.astype(np.uint8)) for img in images.cpu().numpy()]
            batch_loss = 0.0
            for i, (pil_image, ann_text) in enumerate(zip(pil_images, annotation_texts)):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "what is the text in the image?"},
                        ],
                    },
                ]
                text = llm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = llm.processor(text=[text], images=[pil_image], return_tensors="pt").to(llm.model.device)
                labels = llm.processor.tokenizer(ann_text, return_tensors="pt").input_ids.to(llm.model.device)
                # Pad/truncate labels to match input length
                if labels.shape[1] < inputs.input_ids.shape[1]:
                    pad_len = inputs.input_ids.shape[1] - labels.shape[1]
                    labels = torch.cat([labels, torch.full((1, pad_len), -100, dtype=labels.dtype, device=labels.device)], dim=1)
                elif labels.shape[1] > inputs.input_ids.shape[1]:
                    labels = labels[:, :inputs.input_ids.shape[1]]
                # Forward pass with labels for loss computation
                loss = llm.model(input_ids=inputs.input_ids, attention_mask=inputs.get("attention_mask", None), labels=labels).loss
                loss.backward()
                batch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += batch_loss / len(pil_images)
            num_batches += 1
            if iteration % 10000 == 0:
                print(f"Instruction tuning iteration {iteration}, loss={batch_loss / len(pil_images):.4f}")
        print(f"Instruction tuning epoch {epoch+1} average loss: {epoch_loss/num_batches:.4f}")
    print("Instruction tuning complete.")
    # Save the fine-tuned Qwen model
    llm.model.save_pretrained("qwen_finetuned")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # This script will automatically download and use the FUNSD dataset.
    dataset_name = 'funsd'
    split = 'train'  # or 'val', as needed
    download_funsd()
    dataset_root = 'FUNSD/dataset'
    dataset = get_dataset(dataset_name, dataset_root, split)
    num_epochs = 1  # Set number of epochs here
    visual_encoder, mlp = pretrain_stage(dataset, num_epochs=num_epochs)
    llm = QwenVL(device=device)
    instruction_tuning_stage(dataset, visual_encoder, mlp, llm, num_epochs=5)

if __name__ == "__main__":
    main()
