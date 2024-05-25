from llmu.interfaces.gpt_neo_pl import Neo
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, DataLoader
from llmu.utils.mia_validation_ds import MIADataset
from torchmetrics.functional import accuracy
import functools
import tqdm 
import numpy as np 
import re
import transformers
import time
from llmu.utils.plot_utils import save_logl_histograms, save_llr_histograms, save_roc_curves, plot_accuracy_histogram, plot_counterfactual_histogram, plot_scatter_acc_vs_conterfactual
import logging
import math
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import os
import json
from llmu.utils.ufl_utils import convert_numpy_to_list

class NeoMIA(Neo):
    def __init__(self, hparams):
        super(NeoMIA, self).__init__(hparams)

        self.pattern = re.compile(r"<extra_id_\d+>")


        print(f'Loading mask filling model {hparams.mask_model}...')

        n_positions = 512
        
        self.mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(hparams.mask_model, 
                                                                            #  **int8_kwargs, **half_kwargs, 
                                                                             cache_dir=hparams.cache_dir)
        self.mask_tokenizer = transformers.AutoTokenizer.from_pretrained(hparams.mask_model, 
                                                                         model_max_length=n_positions, 
                                                                         cache_dir=hparams.cache_dir)
        
        self.perturbed_decoded = None

        try: 
            self.mask_model.cpu()
        except NameError:
            print("moving mask model to CPU failed due to name error")
            pass 

    def _step(self, batch):
        lm_labels = batch["target_ids"]

        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        loss, score = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels
        )
        
        return loss, score

    def training_step(self, batch, batch_idx):
        loss, score = self._step(batch)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self.hparams.negative_loss:
            return loss * -1
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=-1):
        if self.mode == 'general_lm_eval':

            self.validation_general_lm(batch)
        elif self.mode == 'unlearn':
            value_dict = {}

            # if dataloader_idx in [self.target_validation_idx, -1]:
            if batch['type_path'][0] == 'target':

                value_dict =  self.validation_forget(batch)

            if batch['type_path'][0] == 'target' or batch['type_path'][0] == 'mia':
                og_ll, pert_mean_ll, pert_std_ll, ll_dist = self.perturb_and_compute(batch)

                value_dict['og_ll'] = og_ll
                value_dict['pert_mean_ll'] = pert_mean_ll
                value_dict['pert_std_ll'] = pert_std_ll
                value_dict['ll_dist'] = ll_dist

                return value_dict
            
            # this should restore the legacy flow
            self.validation_general_lm(batch)

        else:
            raise Exception(
                f'Currently not supporting {self.mode}')

    def validation_forget(
            self,
            batch,
            dataset_name='target'):
        loss_reduced, score = self._step(batch)
        self.log(
            'val_loss',
            loss_reduced,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True)

        # This function also measures results of individual examples
        # i.e. results that are not reduced along batch dim
        value_dict = {}

        # MA
        preds, labels = self.validation_ma(batch, dataset_name)
        accs = []
        if len(preds.shape) == 1:
            preds = torch.unsqueeze(preds, 0)
            labels = torch.unsqueeze(labels, 0)

        for pred, label in zip(preds, labels):
            try:
                acc = accuracy(pred, label, ignore_index=-100)
                accs.append(acc)
            except IndexError:
                pass
        if accs:
            accs = torch.stack(accs)
        value_dict['acc'] = accs

        # results = self.perturb_and_compute(batch, dataset_name)
        # value_dict['mia_results'] = results

        return value_dict

    def count_masks(self, texts):
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

    def apply_extracted_fills(self, masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
        tokens = [x.split(' ') for x in masked_texts]

        n_expected = self.count_masks(masked_texts)

        # replace each mask token with the corresponding fill
        for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
            if len(fills) < n:
                tokens[idx] = []
            else:
                for fill_idx in range(n):
                    text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

        # join tokens back into text
        texts = [" ".join(x) for x in tokens]
        return texts
    
    # replace each masked span with a sample from T5 mask_model
    def replace_masks(self, texts):
       
        mask_top_p = 1.0

        n_expected = self.count_masks(texts)
        stop_id = self.mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]

        tokens = self.mask_tokenizer(texts, return_tensors="pt", padding=True).to(self.device)

        outputs = self.mask_model.generate(**tokens, max_length=200, do_sample=True, top_p=mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
        return self.mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def extract_fills(self, texts):
        # remove <pad> from beginning of each text
        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

        # return the text in between each matched mask token
        extracted_fills = [self.pattern.split(x)[1:-1] for x in texts]

        # remove whitespace around each fill
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

        return extracted_fills

    def tokenize_and_mask(self, text, span_length, pct, ceil_pct=False):

        buffer_size = 1
        
        tokens = text.split(' ')
        mask_string = '<<<mask>>>'
        
        span_length = min(int(pct*len(tokens)),span_length)
        #avoid div zero:

        span_length = max(1, span_length)

        n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
        if ceil_pct:
            n_spans = np.ceil(n_spans)
        n_spans = int(n_spans)

        n_masks = 0
        while n_masks < n_spans:
            start = np.random.randint(0, max(1,len(tokens) - span_length))
            end =  start + span_length
            search_start = max(0, start - buffer_size)
            search_end = min(len(tokens), end + buffer_size)
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1
        
        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = ' '.join(tokens)
        return text
    
    def perturb_texts_(self, texts, span_length, pct, ceil_pct=False):


        masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
        raw_fills = self.replace_masks(masked_texts)
        extracted_fills = self.extract_fills(raw_fills)
        perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [self.tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = self.replace_masks(masked_texts)
            extracted_fills = self.extract_fills(raw_fills)
            new_perturbed_texts = self.apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
        
        return perturbed_texts
    
    def perturb_texts(self, texts, span_length, pct, ceil_pct=False):

        chunk_size = 20
        # if '11b' in mask_filling_model_name:
        #     chunk_size //= 2

        outputs = []
        for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
            outputs.extend(self.perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
        return outputs

    def perturb_and_compute(self, batch, n_perturbations = 25):
        

        decoded_batch = [self.tokenizer.decode(row, skip_special_tokens=True) for row in batch['source_ids']]
        # decoded_batch = self.model.batch_decode(batch['source_ids'], skip_special_tokens=True)

        if self.perturbed_decoded is None:
            self.load_mask_model()

            perturb_function = functools.partial(self.perturb_texts, span_length=2, pct=0.3, ceil_pct=False)
            self.perturbed_decoded = perturb_function([x for x in decoded_batch for _ in range(n_perturbations)])

            self.load_base_model()
        
        _parts = []
        for idx in range(len(decoded_batch)):
            _parts.append({
                "original": decoded_batch[idx],
                "perturbed": self.perturbed_decoded[idx * n_perturbations: (idx + 1) * n_perturbations]
            })
        
        og_ll = []
        pert_mean_ll = []
        pert_std_ll = []
        ll_dist = []
        for _part in tqdm.tqdm(_parts, desc="Computing log likelihoods"):
            # print(_part)
            original_loglikelihood = self.compute_likelihood_for_parts(_part["original"])
            perturbed_likelihood_all = torch.stack([self.compute_likelihood_for_parts(perturbed_seq) for perturbed_seq in _part["perturbed"]])
            
            og_ll.append(original_loglikelihood)
            pert_mean_ll.append(torch.mean(perturbed_likelihood_all))
            pert_std_ll.append(torch.std(perturbed_likelihood_all) if len(perturbed_likelihood_all) > 1 else torch.tensor(1))
            
            ll_dist.append(torch.abs(original_loglikelihood - torch.mean(perturbed_likelihood_all)))




        og_ll = torch.stack(og_ll)
        pert_mean_ll = torch.stack(pert_mean_ll)
        pert_std_ll = torch.stack(pert_std_ll)
        ll_dist = torch.stack(ll_dist)

        return og_ll, pert_mean_ll, pert_std_ll, ll_dist

    
    def compute_likelihood_for_parts(self, part):

        length = 200
        
        source = self.tokenizer(
            part,
            max_length=length,
            padding='max_length',
            truncation=True,
            return_tensors="pt").to(self.device)

        targets = self.tokenizer(
            part,
            max_length=length,
            add_special_tokens=False,
            padding='max_length',
            truncation=True,
            return_tensors="pt").to(self.device)
        
        source_ids = source["input_ids"]
        target_ids = targets["input_ids"]
        src_mask = source["attention_mask"]
        target_mask = targets["attention_mask"]

        part_batch = {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask
        }

        loss, _ = self._step(part_batch)
        return -loss

    def load_base_model(self):
        print('MOVING BASE MODEL TO GPU...', end='', flush=True)
        start = time.time()
        try:
            self.mask_model.cpu()
        except NameError:
            print("failed moving mask model to cpu")
            pass
        # if args.openai_model is None:
        self.model.to(self.device)
        print(f'DONE ({time.time() - start:.2f}s)')

    def load_mask_model(self):
        print('MOVING MASK MODEL TO GPU...', end='', flush=True)
        start = time.time()

        self.model.cpu()

        self.mask_model.to(self.device)
        print(f'DONE ({time.time() - start:.2f}s)')


    def validation_epoch_end(self, output):
        if self.hparams.mode in ['unlearn']:

            save_dir = f'llmu_results/{self.hparams.wandb_run_name}'

            if self.init_validation:
                log_col_name = f'init'
            else:
                log_col_name = f'{self.current_epoch:02d}'

            # reduce all outputs from GPUs if multiple
            # if len(self.hparams.valid_sets) > 1:
            #     outputs = self.all_gather(output)[self.target_validation_idx]
            # else:

            outputs = self.all_gather(output)
            
            # wrap it here so logic works for 2 sets
            if len(self.hparams.valid_sets) == 1:
                outputs = [outputs]

            full_output = []
            for idx in range(len(self.hparams.valid_sets)): 
                try:
                    
                    keys = outputs[idx][0].keys()  # [doc_id, loss, acc, el]
                    full_output.append({k: [] for k in keys})

                    # gather all outputs
                    for out in outputs[idx]:
                        for k in keys:
                            full_output[idx][k].append(torch.flatten(out[k]))

                    # refactor into pandas favorable format
                    for k in keys:
                        full_output[idx][k] = torch.cat(full_output[idx][k])
                        full_output[idx][k] = torch.flatten(full_output[idx][k]).cpu().numpy()
                except Exception as e:
                    print(f"Error: {e}")
                    pass
            # outputs = outputs[0]
            # check histograms for these two
            if len(full_output) == 1:
                # if self.init_validation:
                plot_accuracy_histogram(full_output[0]['acc'], save_dir, log_col_name)
                plot_counterfactual_histogram(full_output[0]['ll_dist'], save_dir, log_col_name)
                plot_scatter_acc_vs_conterfactual(full_output[0]['acc'], full_output[0]['ll_dist'], save_dir, log_col_name)

            if len(full_output) == 2:
                assert len(full_output[0]['ll_dist']) == len(full_output[1]['ll_dist']), "Inconsistent output between target and mia length!!!"

                save_logl_histograms(full_output[0], full_output[1], save_dir, log_col_name)
                save_llr_histograms(full_output[0], full_output[1], save_dir, log_col_name)

                fpr, tpr, roc_auc = self.get_roc_metrics(full_output[1]['ll_dist'], full_output[0]['ll_dist'])
                
                save_roc_curves(fpr, tpr, roc_auc, save_dir, log_col_name)

                p, r, pr_auc = self.get_precision_recall_metrics(full_output[1]['ll_dist'], full_output[0]['ll_dist'])
            

            ma = np.mean(full_output[0]['acc'])
            # el = df[f'el_{self.el_n_main}-gram_{log_col_name}'].mean()
            if self.current_epoch >= self.hparams.min_train_epochs:

                if ma < self.hparams.ma_threshold: # and el < self.hparams.el_threshold:
                    logging.info(
                        f'Early Stopping as Forgetting Threshold is reached, {ma=}')
                    self.trainer.should_stop = True


            # convert numpy to list 

            file_path = f'{save_dir}/val_data_{log_col_name}.json'
            array_out = convert_numpy_to_list(full_output)
            with open(file_path, 'w') as json_file:
                json.dump(array_out, json_file, indent=2) 

    def on_validation_end(self):
        self.init_validation = False

    def on_validation_start(self):
        save_dir = f'llmu_results/{self.hparams.wandb_run_name}'
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)

    def get_roc_metrics(self, real_preds, sample_preds):
        real_preds =  [element for element in real_preds if not math.isnan(element)]
        sample_preds = [element for element in sample_preds if not math.isnan(element)]
        fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
        roc_auc = auc(fpr, tpr)
        return fpr.tolist(), tpr.tolist(), float(roc_auc)

    def get_precision_recall_metrics(self, real_preds, sample_preds):
        real_preds =  [element for element in real_preds if not math.isnan(element)]
        sample_preds = [element for element in sample_preds if not math.isnan(element)]

        precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
        pr_auc = auc(recall, precision)
        return precision.tolist(), recall.tolist(), float(pr_auc)

    def on_train_end(self):
        pass


    def get_training_dataset(self, dataset_name, tokenizer,
                    valid_subset_path, type_path, length=None):
        input_length = length if length else self.hparams.input_length
        output_length = length if length else self.hparams.output_length
        dataset = MIADataset(
            # TODO: add local directory to enron dataset: https://www.cs.cmu.edu/~enron/
            data_dir="<obfuscated>",
            cache_dir=self.hparams.cache_dir,
            tokenizer=tokenizer,
            type_path=type_path,
            input_length=input_length,
            output_length=output_length,
            dataset_name=dataset_name,
            args=self.hparams)
        return dataset


    def train_dataloader(self):
        dataset = self.hparams.train_set
        length = None
        if self.mode == 'unlearn':
            length = self.target_length

        # NOTE: only enron implemented for now
        train_dataset = self.get_training_dataset(
            dataset_name=dataset,
            tokenizer=self.tokenizer,
            valid_subset_path="",
            type_path="train",
            length=length)

        sampler = RandomSampler(train_dataset)
        dataloader = DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers)
        return dataloader
    

    def val_dataloader(self):
        datasets = []
        target_idx = -1
        for i in range(len(self.hparams.valid_sets)):
            dataset = self.hparams.valid_sets[i]
            valid_subset_path = self.hparams.valid_subset_path[i]
            type_path = self.hparams.valid_type_path[i]
            dataset_name = dataset

            length = None
            if type_path == 'target' or type_path == 'mia':
                length = self.target_length

            dataset = self.get_training_dataset(
                dataset_name=dataset_name,
                tokenizer=self.tokenizer,
                valid_subset_path="",
                type_path=type_path,
                length=length)
            
            datasets.append(dataset)

        # # Setup the dataframe for logging MA and EL of individual examples
        # if self.mode in ['unlearn'] and self.valid_df is None:
        #     target_idx = self.hparams.valid_type_path.index('target')
        #     self.target_validation_idx = target_idx
        #     self.valid_df = datasets[target_idx].dataset
        #     self.valid_df = self.valid_df.set_index('doc_id')
        #     self.valid_df_index = self.valid_df.index
        #     # The reference prefix for logging Table 3.
        #     self.valid_df['prefix'] = self.valid_df['text'].apply(
        #         lambda x: self.tokenizer.decode(self.tokenizer.encode(x)[:100]))

        dataloaders = []
        for i, dataset in enumerate(datasets):
            # change this for target and mia
            if self.mode in ['unlearn'] and dataset in ['target', 'mia']:
                # For the unlearning target data, match the eval batch_size to
                # train batch_size
                batch_size = self.hparams.train_batch_size * \
                    self.hparams.gradient_accumulation_steps
                
                dataloaders.append(
                    DataLoader(
                        dataset,
                        batch_size=batch_size,
                        num_workers=self.hparams.num_workers,
                        shuffle=False))
            else:
                # For other evaluation datasets
                dataloaders.append(
                    DataLoader(
                        dataset,
                        batch_size=self.hparams.eval_batch_size,
                        num_workers=self.hparams.num_workers,
                        shuffle=False))
        return dataloaders