import torch
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, GPT2Tokenizer
from deepspeed.ops.adam import DeepSpeedCPUAdam
from peft import LoraConfig, get_peft_model, TaskType
from deepspeed.accelerator import get_accelerator
from torch.utils.data import RandomSampler, DataLoader
from llmu.utils.dataset_utils import UFLDataset
import deepspeed
from accelerate.hooks import AlignDevicesHook

def remove_hook_from_module(module: torch.nn.Module, recurse=False, hook_cls=AlignDevicesHook):

    if hasattr(module, "_hf_hook") and isinstance(module._hf_hook, hook_cls):
        module._hf_hook.detach_hook(module)
        delattr(module, "_hf_hook")

        if hasattr(module, "_old_forward"):
            module.forward = module._old_forward
            delattr(module, "_old_forward")

    if recurse:
        for child in module.children():
            remove_hook_from_module(child, recurse)

    return module


class PeftCausal(pl.LightningModule):
    def __init__(self, hparams):
        super(PeftCausal, self).__init__()
        self.mode = hparams.mode

        # Model Initializaion
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            hparams.tokenizer_name_or_path)
        if 'gpt' in hparams.tokenizer_name_or_path:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            hparams.model_name_or_path,
            resid_dropout=0,
            embed_dropout=0,
            attention_dropout=0,
            pad_token_id=self.tokenizer.eos_token_id)
        # self.model = AutoModelForCausalLM.from_pretrained(hparams.model_name_or_path, 
        #                                                   pad_token_id=self.tokenizer.eos_token_id,
        #                                                   trust_remote_code=True, 
        #                                                 #   device_map="auto",
        #                                                   quantization_config=hparams.quantization_config)
                                                        #   load_in_8bit=load_in_8bit)

        if hparams.peft:
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.00)
            self.model = get_peft_model(self.model, peft_config)
        
        # remove_hook_from_module(self.model, recurse=True)

        self.save_hyperparameters(hparams)
        # I don't think this is needed
        # self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.target_length = self.hparams.target_length if self.hparams.target_length else self.hparams.input_length

        self.target_validation_idx = None
        # Flag to check wheter this is the initial validation before training
        self.init_validation = True


    def forward(self, input_ids, attention_mask=None, lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
        )
        
    def _step(self, batch):
        lm_labels = batch["target_ids"]

        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(

            input_ids = batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels
        )
        loss, score = outputs[0], outputs[1]
        return loss, score
    
    def training_step(self, batch, batch_idx):

        get_accelerator().empty_cache()
        torch.cuda.empty_cache()

        loss, score = self._step(batch)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self.hparams.negative_loss:
            return loss * -1
        return loss
    
    # def training_step(self, batch, batch_idx):
    #     get_accelerator().empty_cache()
    #     input_ids, target_start_idx = batch
    #     logits = self.model(input_ids).logits
    #     loss = mt_loss(logits, input_ids, target_start_idx, self.hparams.pad_token_id)
    #     self.log("train_loss", loss)
    #     return loss
    def validation_step(self, batch, batch_idx, dataloader_idx=-1):
        if self.mode == 'unlearn':
            if dataloader_idx in [self.target_validation_idx, -1]:
                # return self.validation_forget(batch)
                self.validation_forget(batch)
        else: 
            raise Exception(f'Currently not supporting {self.mode}')\
            
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
        

    def get_dataset(self, dataset_name, tokenizer,
                    valid_subset_path, type_path, length=None):
        input_length = length if length else self.hparams.input_length
        output_length = length if length else self.hparams.output_length
        dataset = UFLDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            valid_subset_path=valid_subset_path,
            type_path=type_path,
            input_length=input_length,
            output_length=output_length,
            args=self.hparams)
        return dataset

    def train_dataloader(self):
        dataset = self.hparams.train_set
        length = None
        if self.mode == 'unlearn':
            length = self.target_length

        train_dataset = self.get_dataset(
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
            if type_path == 'target':
                length = self.target_length

            dataset = self.get_dataset(
                dataset_name=dataset_name,
                tokenizer=self.tokenizer,
                valid_subset_path=valid_subset_path,
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
            if self.mode in ['unlearn'] and i == target_idx:
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
    # def validation_step(self, batch, batch_idx):
    #     input_ids, target_start_idx = batch
    #     logits = self.model(input_ids).logits
    #     loss = mt_loss(logits, input_ids, target_start_idx, self.hparams.pad_token_id)
    #     self.log("val_loss", loss, sync_dist=True)
    #     self.log("val_ppl", torch.exp(loss), sync_dist=True)
    #     return loss

    # def generate(self, batch, **kwargs):
    #     return self.model.generate(batch,**kwargs)
    
    # def configure_optimizers(self):
    #     return 
    
    def configure_optimizers(self):
        parameters = self.model.parameters()
        if self.hparams.strategy in ['deepspeed_stage_2']:
            optimizer = deepspeed.ops.adam.FusedAdam(
                parameters,
                lr=self.hparams.learning_rate,
                betas=(0.9, 0.98))
        elif self.hparams.strategy in ['deepspeed_stage_2_offload']:
            optimizer = DeepSpeedCPUAdam(parameters, 
                                lr=self.hparams.learning_rate, 
                                # weight_decay=self.hparams.weight_decay, 
                                betas=(0.9, 0.98))
        else:
            optimizer = torch.optim.Adam(
                parameters,
                lr=self.hparams.learning_rate,
                betas=(0.9, 0.98))
        return [optimizer]