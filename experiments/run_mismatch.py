
import argparse
import logging
import random
import time
from llmu.interfaces.gpt_neo_mismatch import neo_mismatch
import os
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    cache_dir = 'cache'
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")


    parser.add_argument("--use_lora", action="store_true", default=True)

    parser.add_argument(
        "--max_unlearn_steps",
        type=int,
        default=500,
        help="Max number of unlearning steps.",
    )
    parser.add_argument(
        "--bad_weight", type=float, default=0.5, help="Weight on the bad loss."
    )
    parser.add_argument(
        "--random_weight",
        type=float,
        default=1,
        help="Weight on learning the random outputs.",
    )
    parser.add_argument(
        "--normal_weight",
        type=float,
        default=1,
        help="Weight on normal loss.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size of unlearning."
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Unlearning LR.")
    parser.add_argument(
        "--max_bad_loss",
        type=float,
        default=100,
        help="Maximum loss on bad samples to terminate.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/gpt-neo-125M",
        help="Name of the pretrained model.",
    )

    parser.add_argument(
        "--model_save_dir",
        type=str,
        default=f"models/125M_unlearned",
        help="Directory to save model.",
    )
    parser.add_argument(
        "--save_every", type=int, default=500, help="How many steps to save model."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="experiments/data/extraction/lm_extraction_16_0.csv",
        help="Target dataset for unlearning",
    )

    args = parser.parse_args()
    torch.set_float32_matmul_precision('medium')


    # fire the neo mismatch journey
    neo_mismatch(args)