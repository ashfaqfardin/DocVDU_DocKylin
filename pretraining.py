"""
dockylin_pretrain.py
===============================================
DocKylin pre‑training exactly as in the paper:
 • Visual encoder : Donut‑Swin‑Large (0.07 B)  – pretrained
 • LLM            : Qwen‑7B‑Chat (frozen)
 • APS            : Adaptive Pixel Slimming   – both phases
 • Objective      : full‑text recognition  (cross‑entropy)
 ===============================================
Expected input manifest (jsonl):
{ "image": "path/to/img_001.jpg", "text": "full ground‑truth text" }
This file lists ALL datasets in Table 2 of the paper concatenated.
"""

# ---------- Imports ----------
import os, math, json, argparse, cv2, numpy as np, torch
from PIL import Image
from tqdm import tqdm

from aps import crop_image                       # your uploaded APS
import timm                                      # Donut‑Swin
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig)

# ---------- Hyper‑parameters (paper) ----------
MAX_PIXELS          = 1728 * 1728
HID_DIM             = 4096           # Qwen‑7B‑Chat hidden size
BATCH_SIZE_DEFAULT  = 8
LR_START, LR_END    = 1e-4, 1e-5

# ---------- APS ----------
def apply_aps(img_bgr):
    cropped, *_ = crop_image(img_bgr, direction="xy")
    h, w = cropped.shape[:2]
    if h * w > MAX_PIXELS:
        scale = (MAX_PIXELS / (h * w)) ** 0.5
        cropped = cv2.resize(cropped, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    return cropped

# ---------- Dataset ----------
class JSONLDocDataset(torch.utils.data.Dataset):
    def __init__(self, manifest):
        self.samples = [json.loads(l) for l in open(manifest, encoding="utf-8")]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(item["image"]), cv2.COLOR_BGR2RGB)
        img = apply_aps(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return img, item["text"]

def collate_fn(batch):
    imgs, txts = zip(*batch)
    return list(imgs), list(txts)

# ---------- Vision Encoder ----------
class DonutSwinEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_large_patch4_window12_384",
            pretrained=True, features_only=True, out_indices=(3,)
        )                                          # C=1024
        self.proj = torch.nn.Linear(1024, HID_DIM)

    def forward(self, imgs):
        x = (torch.as_tensor(imgs, dtype=torch.float32).permute(0,3,1,2)/255)
        x = torch.nn.functional.interpolate(x, size=(384,384),
                                            mode="bilinear", align_corners=False)
        feat = self.backbone(x)[-1]               # (B,C,H,W)
        feat = feat.flatten(2).transpose(1,2)     # (B,N,C)
        return self.proj(feat)                    # (B,N,4096)

# ---------- Qwen‑7B‑Chat (frozen) ----------
def load_frozen_qwen(dtype=torch.float16):
    bnb = BitsAndBytesConfig(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-7B-Chat", device_map="auto",
        quantization_config=bnb, torch_dtype=dtype
    )
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat")
    tok.pad_token = tok.eos_token
    model.eval(); model.gradient_checkpointing_enable()
    for p in model.parameters(): p.requires_grad=False
    return model, tok

# ---------- Prefix fusion (LLaVA‑style) ----------
def build_inputs(tok, img_emb, gt_text, device):
    """
    <img><im_patch>*N ###OCR\n GT_TEXT<eos>
    image embeddings replace <im_patch> token embeddings.
    """
    N = img_emb.size(1)
    prompt = "<img>" + "<im_patch>"*N + " ###OCR\n"
    ids = tok(prompt, add_special_tokens=False).input_ids
    label_ids = tok(gt_text, add_special_tokens=False).input_ids + [tok.eos_token_id]

    input_ids  = torch.tensor(ids + label_ids, device=device).unsqueeze(0)
    labels     = torch.tensor([-100]*len(ids) + label_ids, device=device).unsqueeze(0)

    # create placeholder embeddings & then replace
    inputs_emb = model.model.embed_tokens(input_ids)
    start = ids.index(tok.convert_tokens_to_ids("<im_patch>"))
    inputs_emb[0, start:start+N, :] = img_emb[0]

    return dict(inputs_embeds=inputs_emb, labels=labels)

# ---------- Cosine LR ----------
def cosine_lr(step, total):
    return LR_END + 0.5*(LR_START-LR_END)*(1+math.cos(math.pi*step/total))

# ---------- Training Loop ----------
def train(args):
    ds = JSONLDocDataset(args.data_manifest)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                     shuffle=True, collate_fn=collate_fn,
                                     num_workers=4, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vision = DonutSwinEncoder().to(device)
    optim  = torch.optim.AdamW(vision.parameters(), lr=LR_START)

    global model, tok   # used in build_inputs()
    model, tok = load_frozen_qwen()
    total_steps = args.max_iters
    step = 0

    pbar = tqdm(total=total_steps, desc="Pre‑training")
    while step < total_steps:
        for imgs, txts in dl:
            if step >= total_steps: break
            step += 1; pbar.update(1)

            imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs]
            img_emb = vision(np.stack(imgs)).to(device)  # (B,N,4096)

            loss = 0
            for i in range(len(imgs)):
                batch_emb = img_emb[i:i+1]               # (1,N,4096)
                inputs = build_inputs(tok, batch_emb, txts[i], device)
                out = model(**inputs)
                loss += out.loss
            loss = loss / len(imgs)

            loss.backward()
            lr = cosine_lr(step, total_steps)
            for g in optim.param_groups: g["lr"]=lr
            optim.step(); optim.zero_grad()

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.6f}")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(vision.state_dict(), os.path.join(args.output_dir, "vision.pth"))
    print("✅ Finished 450 k‑step pre‑training. Weights saved to", args.output_dir)

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_manifest", type=str, required=True,
                        help="jsonl manifest containing {image, text}")
    parser.add_argument("--output_dir",   type=str, default="ckpt_pretrain")
    parser.add_argument("--batch_size",   type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--max_iters",    type=int, default=450_000)
    args = parser.parse_args()
    train(args)
