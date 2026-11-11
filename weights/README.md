# Pretrained weights

Place the four ImageNet-pretrained timm checkpoints here as `.safetensors` files:

    python scripts/download_weights.py --out weights
    export RMBFF_WEIGHTS=$PWD/weights

SHA-256 checksums of the files used in the paper are listed in `outputs/weights_sha256.txt`.
