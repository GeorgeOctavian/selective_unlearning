# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
A script to show an example of how to unlearn harmfulness.

The dataset used in is `PKU-SafeRLHF`. Model support OPT-1.3B, OPT-2.7B, and Llama 2 (7B).
"""
import argparse
import logging
import random
import time
import pandas as pd
import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from torchmetrics.functional import accuracy
from peft import AdaLoraConfig, TaskType, get_peft_model, LoraConfig
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from llmu.utils.mismatch_utils import (
    compute_kl,
    create_pku_dataloader_from_dataset,
    create_truthfulqa_dataloader,
    create_forget_dataloader_from_Pile,
    get_answer_loss,
    get_rand_ans_loss,
    get_rand_complete_loss,
    get_truthfulQA_answers_plaintext,
)

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

def generate(model, **kwargs):
        return model.generate(**kwargs)

def validation_ma(max_len, model, tokenizer, input_ids):
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        labels, preds = [], []
        for i in range(1, max_len):
            label = input_ids[..., i]
            prompt = input_ids[..., :i]
            try:
                pred = generate(model=model, input_ids=prompt, max_length=i + 1)[:, -1]
            except IndexError:  # if batch == 1
                pred = generate(model=model, input_ids=torch.squeeze(
                    prompt), max_length=i + 1).squeeze()[-1]

            labels.append(torch.squeeze(label))
            preds.append(torch.squeeze(pred))
        
        preds = torch.stack(preds)
        labels = torch.stack(labels)
        score = accuracy(preds, labels, ignore_index=-100)
        return score

def neo_mismatch(args) -> None:
    accelerator = Accelerator()
    # device = accelerator.device
    device=accelerator.device

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                resid_dropout=0,
                embed_dropout=0,
                attention_dropout=0,
                pad_token_id=tokenizer.eos_token_id)
    
    # If use LoRA.
    if args.use_lora:
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=512,
            lora_alpha=1024,
            # target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, peft_config)

    model.to(device)


    # Load forget data
    lm_data = pd.read_csv(args.dataset_name, lineterminator='\n')
    forget_dataset = Dataset.from_pandas(lm_data)

    parts = args.dataset_name.split('/')

    # Extract the part containing the filename
    filename_part = parts[-1]

    # Remove the ".csv" extension
    filename_without_extension = filename_part.split('.')[0]

    train_bad_loader = create_forget_dataloader_from_Pile(
        tokenizer, forget_dataset, batch_size=args.batch_size
    )

    # Get normal data.
    train_normal_loader, _, _ = create_truthfulqa_dataloader(
        tokenizer, batch_size=args.batch_size
    )

    # Load normal answer used for random mismatch.
    normal_ans = get_truthfulQA_answers_plaintext()

    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Prepare.
    num_training_steps = args.max_unlearn_steps
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    (
        model,
        optimizer,
        train_bad_loader,
        train_normal_loader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_bad_loader, train_normal_loader, lr_scheduler
    )


    # Reference model for computing KL.
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                            pad_token_id=tokenizer.eos_token_id)
    pretrained_model.to(device)

    # Start unlearning.
    bad_loss = 0.0
    idx = 0
    start_time = time.time()
    max_ma = 0.32
    current_ma = 1.
        
    # Stop if bad loss is big enough or reaching max step.
    while (max_ma - current_ma) < 0.0001 and idx < args.max_unlearn_steps:

        model.train()
        for bad_batch, normal_batch in zip(train_bad_loader, train_normal_loader):
            ############ GA on answer only. ############
            bad_loss = get_answer_loss(tokenizer, "ga", bad_batch, model, device=device)

            ############ Random mismatch. ############
            random_loss = get_rand_complete_loss(
                bad_batch,
                tokenizer,
                normal_ans,
                model,
                K=5,
                device=device,
            )

            ############ KL on normal samples. ############
            normal_loss = compute_kl(pretrained_model, model, normal_batch, device)

            # Final loss = bad loss + random smoothing + normal loss.
        
            loss = (
                args.bad_weight * bad_loss
                + args.random_weight * random_loss
                + args.normal_weight * normal_loss
            )

            # torch.cuda.empty_cache()
            # Backprop.
            accelerator.backward(loss)
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()

            stats = (
                f"batch: {idx}, "
                f"bad_loss: {-bad_loss:.2f}, "
                f"current_div_loss: {normal_loss:.2f}, "
            )
            logging.info(stats)
            print(stats)
            idx += 1

            # Save model.
            if idx % args.save_every == 0:
                model.save_pretrained(args.model_save_dir, from_pt=True)
        
        # torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():

            ma = validation_ma(200, model, tokenizer, train_bad_loader.dataset['input_ids'])
            current_ma = ma.cpu()

            print(f"Memorisation scores: {current_ma}")
            # torch.cuda.empty_cache()

    end_time = time.time()
    logging.info("Total time: %d sec" % (end_time - start_time))

    if args.use_lora:
        model = model.merge_and_unload()

    # Save final model.
    model.save_pretrained(f"{args.model_save_dir}_{filename_without_extension}", from_pt=True)
    logging.info("Unlearning finished")

    return



