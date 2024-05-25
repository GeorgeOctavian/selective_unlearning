from transformers import GPT2Tokenizer, AutoModelForCausalLM
import pytorch_lightning as pl
import deepspeed
import torch
from torch.utils.data import RandomSampler, DataLoader
from llmu.utils.dataset_utils import UFLDataset
from llmu.utils.llmu_neighbor_ds import NbDataset

class GenericAutoregressiveModule(pl.LightningModule):
    def __init__(self, hparams):
        super(GenericAutoregressiveModule, self).__init__()
        self.mode = hparams.mode

        # Model Initializaion
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            hparams.tokenizer_name_or_path)
        if 'gpt' in hparams.tokenizer_name_or_path:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Different models have different kwargs
        if 'gpt-neo' in hparams.model_name_or_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                hparams.model_name_or_path,
                resid_dropout=0,
                embed_dropout=0,
                attention_dropout=0,
                pad_token_id=self.tokenizer.eos_token_id)
        elif 'opt' in hparams.model_name_or_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                hparams.model_name_or_path, dropout=0, attention_dropout=0, activation_dropout=0)
        else:  # GPT2
            self.model = AutoModelForCausalLM.from_pretrained(
                hparams.model_name_or_path,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
                pad_token_id=self.tokenizer.eos_token_id)

        self.save_hyperparameters(hparams)

        # is this needed?
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.target_length = self.hparams.target_length if self.hparams.target_length else self.hparams.input_length

    def forward(self, input_ids, attention_mask=None, lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
        )


    def _step(self, batch):
        # is this too hard-core?
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels
        )
        loss, score = outputs[0], outputs[1]
        return loss, score


    def training_step(self, batch, batch_idx):
        loss, _ = self._step(batch)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        print(f'The loss is {loss}')
        return loss
    
    
    def configure_optimizers(self):
        parameters = self.model.parameters()
        if self.hparams.strategy in ['deepspeed_stage_2']:
            optimizer = deepspeed.ops.adam.FusedAdam(
                parameters,
                lr=self.hparams.learning_rate,
                betas=(0.9, 0.98))
        else:
            optimizer = torch.optim.Adam(
                parameters,
                lr=self.hparams.learning_rate,
                betas=(0.9, 0.98))
        return [optimizer]


    def validation_step(self, batch, batch_idx):
        pass
        # loss_reduced, _ = self._step(batch)
        # self.log(
        #     'val_loss',
        #     loss_reduced,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        #     add_dataloader_idx=False,
        #     sync_dist=True)

    # Additional hooks that you might find useful
    # def training_epoch_end(self, outputs):
    #     pass

    # def validation_epoch_end(self, outputs):
    #     pass

    # def on_epoch_start(self):
    #     pass

    # def on_epoch_end(self):
    #     pass

    # def on_train_start(self):
    #     pass

    # def on_train_end(self):
    #     pass


    # keep this intact for now
    def get_dataset(self, dataset_name, tokenizer,
                    valid_subset_path, type_path, length=None):
        input_length = length if length else self.hparams.input_length
        output_length = length if length else self.hparams.output_length
        
        learning_task = "nb"    
        if learning_task == "nb":
            current_device = torch.cuda.current_device() 
            dataset = NbDataset(tokenizer=tokenizer, 
                dataset_name=dataset_name,
                valid_subset_path=valid_subset_path,
                type_path=type_path, 
                input_length=input_length,
                output_length=output_length,
                device=self.device,
                args=self.hparams)
        else: 
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


        if self.device.type == 'cpu':
            return None

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

        learning_task = "nb" 

        if learning_task == "nb":
            # batch_size = self.hparams.n_perturbations 
            dataloader = DataLoader(
                train_dataset,
                sampler=sampler,
                batch_size=self.hparams.train_batch_size,
                num_workers=self.hparams.num_workers)
        return dataloader