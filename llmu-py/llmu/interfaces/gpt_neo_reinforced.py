from typing import Optional
from datasets import load_metric
import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import torch
import time
from tqdm import tqdm
import pandas as pd
tqdm.pandas()
from multiprocessing import Pool
from datasets import load_dataset, Dataset
from evaluate import load
from tqdm import tqdm
import os
import os
# set visible devices here, for the device_map to work


class RLNeo():

    def __init__(self, hparams):
        super(RLNeo, self).__init__()

        self.sacrebleu = load_metric('sacrebleu')
        self.bertscore = load("bertscore")

        # Model Initializaion
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            hparams['lm_name'])
        if 'gpt' in hparams['lm_name']:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            hparams['lm_name'],
            device_map="auto")

        self.model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(
            hparams['lm_name'], 
            device_map="auto")

        self.ppl_model = AutoModelForCausalLM.from_pretrained(
            hparams['lm_name'], 
            device_map="auto")    

        # self.save_hyperparameters(hparams)
        self.hparams = hparams

        # self.model.resize_token_embeddings(len(self.tokenizer))
        # self.target_length = self.hparams.target_length if self.hparams.target_length else self.hparams.input_length
        


    def sacrebleu_fn(self, label, response):
        score = self.sacrebleu.compute(predictions=[response], references=[[label]])['score']
        return 100-score 

    def reward_sacrbleu(self, response, label):
        pool = Pool()
        result = [pool.apply(self.sacrebleu_fn, args=(true, pred)) for true, pred in zip(label, response)]
        return result

    def reward_fn(self, response, label, label_texts, generated_texts):
    
        # bleu_score = self.reward_sacrbleu(response, label)
        bleu_score = 0    
        # ppl_lb_tns, ppl_lb_mean = perplexity_fn(label_texts)

        # ppl_gen_tns, ppl_gen_mean = perplexity_fn(generated_texts)
        
        score = self.reward_fn_comp(label, response)
        
        if score == 'bleu':
            score = bleu_score
        
        return score, bleu_score
    
    # this is the reward function
    def reward_fn_comp(self, label, response):
        if self.hparams['reward_type'] == 'bleu':
            return 'bleu'
        
        if self.hparams['reward_type'] == 'bert':
            score = self.bertscore.compute(predictions=response, references=label, 
                                model_type="microsoft/deberta-large", device='cuda')['f1']
            score = [-abs(number) for number in score]
            
        return score

    def load_dataset(self):

        def tokenize(sample):

            source = self.tokenizer.encode(sample['text'])
            
            # first 100 tokens
            sample["prefix_tokens"] = source[:50]
            # last 100 tokens
            sample["suffix_tokens"] = source[50:] 

            sample['query'] = self.tokenizer.decode(sample['prefix_tokens'])
            sample['suffix'] = self.tokenizer.decode(sample['suffix_tokens'])
            return sample 

        lm_data = pd.read_csv(self.hparams['dataset_name'], lineterminator='\n')
        ds = Dataset.from_pandas(lm_data)

        return ds.map(tokenize, batched=False)

    def ppo_train(self, ppo_config, total_ppo_epochs):
        
        collater = lambda data: dict((key, [d[key] for d in data]) for key in data[0])
        
        gen_kwargs = {
            "min_length":-1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=ppo_config.learning_rate)
        # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
        ppo_trainer = PPOTrainer(
            ppo_config, self.model, ref_model=self.model_ref, tokenizer=self.tokenizer, dataset=self.load_dataset(), data_collator=collater, optimizer=optimizer
        )

        # for _ in tqdm(total_ppo_epochs):
            # ideally we should shuffle here; but let's do it iteratively
        # for _, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        for epoch, batch in tqdm(zip(range(total_ppo_epochs), iter(ppo_trainer.dataloader))):

            logs, timing = dict(), dict()
            t0 = time.time()
            query_tensors = [torch.tensor(t).long().cuda() for t in batch["prefix_tokens"]]

            #### Get response from lm
            t = time.time()
            response_tensors = []
            for i in range(ppo_trainer.config.batch_size):
                gen_len = 150
                # query_tensor_sq = query_tensors[i].unsqueeze(dim=0)
                gen_kwargs["max_new_tokens"] = gen_len
                response = ppo_trainer.generate(query_tensors[i], **gen_kwargs)

                response_tensors.append(response.squeeze()[-gen_len:])

            batch['response'] = [self.tokenizer.decode(r.squeeze()) for r in response_tensors]
            timing['time/get_response'] = time.time()-t
            #### Compute reward score
            t = time.time()
            respones_batch = batch['response']
            label_batch = batch['suffix']

            label_texts = [q + r for q,r in zip(batch['query'], batch['suffix'])]
            generated_texts = [q + r for q,r in zip(batch['query'], batch['response'])]

            reward_scores, bleu_score = self.reward_fn(respones_batch, label_batch, label_texts, generated_texts)
            rewards = torch.tensor(reward_scores, dtype=float).cuda()
            rewards = [torch.tensor(output) for output in rewards]

            timing['time/get_sentiment_preds'] = time.time()-t
            print('finished reward', rewards)
            #### Run PPO step 
            t = time.time()

            self.model.gradient_checkpointing_enable()
            self.model.pretrained_model.config.use_cache = False


            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)


            timing['time/optimization'] = time.time()-t

            #### Log everything
            timing['time/epoch'] = time.time()-t0
            rewards = torch.tensor(reward_scores, dtype=float)
            # table_rows = [list(r) for r in zip(batch['query'], batch['response'], rewards.cpu().tolist(), batch['suffix'], bleu_score)]

            # logs.update({'game_log': wandb.Table(columns=['query', 'pred', 'reward','label', 'bleu_score'], rows=table_rows)})
            # logs.update(timing)
            # logs.update(stats)
            logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
            logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
            # logs['env/perplexity_gen'] = mean_ppl_gen.cpu().numpy()
            # logs['env/perplexity_lab'] = mean_ppl_label.cpu().numpy()
            # logs['env/bleu'] = statistics.mean(bleu_score)
            logs['env/reward_dist'] = rewards.cpu().numpy()

            ppo_trainer.accelerator.log(logs)

            # self.model.save_pretrained('saved_models/', max_shard_size='20GB')