
import torch
import os

path = r"c:\Users\Long\Documents\Hoc_Tap\SE363 (1)\model\bert_absa_model.pth"
if not os.path.exists(path):
    print("File not found")
    exit(1)

try:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict):
        print("Keys:", list(checkpoint.keys()))
    else:
        print("Not a dict, type:", type(checkpoint))
except Exception as e:
    print("Error:", e)
