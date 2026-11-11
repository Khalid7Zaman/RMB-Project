"""Download the four ImageNet-pretrained backbones as .safetensors (run on a machine with internet).

    python scripts/download_weights.py --out weights
Then set RMBFF_WEIGHTS=weights. SHA-256 sums of the files used in the paper are in
outputs/weights_sha256.txt.
"""
import argparse, os, urllib.request
NAMES = ["mobilenetv3_large_100.ra_in1k", "vit_small_patch16_224.augreg_in21k_ft_in1k",
         "regnety_016.tv2_in1k", "seresnet50.a1_in1k"]
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="weights"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
for n in NAMES:
    url = f"https://huggingface.co/timm/{n}/resolve/main/model.safetensors"
    dst = os.path.join(a.out, f"{n}.safetensors")
    if not os.path.exists(dst):
        print("downloading", url); urllib.request.urlretrieve(url, dst)
