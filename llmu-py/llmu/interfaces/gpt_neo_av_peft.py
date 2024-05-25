from llmu.interfaces.gpt_neo_pl import Neo
from llmu.model_edit.arithmetic_edit import TaskVector
from llmu.utils.plot_utils import plot_accuracy_histogram
from transformers import GPT2Tokenizer, AutoModelForCausalLM
import logging
# differential privacy decoding with Neo gpt 
import pandas as pd 
import copy
import torch
from peft import LoraConfig, get_peft_model, TaskType
from deepspeed.accelerator import get_accelerator
import json 
from llmu.utils.ufl_utils import convert_numpy_to_list

class NeoAVP(Neo):
    def __init__(self, hparams):
        super(NeoAVP, self).__init__(hparams)

        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=64, lora_dropout=0.0)
        self.model = get_peft_model(self.model, peft_config)
        
        self.edit_started = False
        self.previous_state_dict = None


    def training_step(self, batch, batch_idx):
        get_accelerator().empty_cache()
        torch.cuda.empty_cache()

        if not self.edit_started:
            indexes = set()
            index_max = None
            max_cur_mem = float('-inf')
            for i, value in enumerate(batch['doc_id']):
                cur_mem = self.mem_per_sample.get(value.item(), self.hparams.mem_threshold)
                if cur_mem >= self.hparams.mem_threshold:
                    indexes.add(i)

                if cur_mem > max_cur_mem:
                    index_max = i
                    max_cur_mem = cur_mem

                
            indexes.add(index_max)

            print(f"We have the following indexes: {indexes} @@@@@@@@")
            
            index_tensor = torch.tensor(list(indexes), device=self.device)
            
            for key, value in batch.items():
                # x_select to filter rows based on indices
                if torch.is_tensor(value):
                    filtered_tensor = torch.index_select(value, dim=0, index=torch.tensor(index_tensor))
                    batch[key] = filtered_tensor

        loss, score = self._step(batch)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # NOTE: this fine-tunes for specific examples, minimise loss instead of maximising
        if self.hparams.negative_loss:
            return loss * -1
        print(f'Training loss is: {loss} @@@@@')
        return loss


    def artm_edit_before_val(self, current_state_dict):
        assert self.hparams.first_coef < 0, "first coefficient should be negative"

        with torch.no_grad():
            # Convert current_state_dict to CPU
            current_state_dict_cpu = {key: value.cpu() for key, value in current_state_dict.items()}

            # Compute task drift without unnecessary GPU operations
            unlearnt_dict = {}

            for key in current_state_dict_cpu:

                current_state_dict_cpu[key] = current_state_dict_cpu[key] - self.original_state_dict[key]
                if "lora_B" in key:
                    # subtract lora B 
                    unlearnt_dict[key] = self.original_state_dict[key] + self.hparams.first_coef * current_state_dict_cpu[key]
                elif "lora_A" in key: 
                    # add lora A so then we have
                    unlearnt_dict[key] = self.original_state_dict[key] + current_state_dict_cpu[key]
                else:
                    unlearnt_dict[key] = self.original_state_dict[key]

            # Load the new state_dict directly without unnecessary GPU operations
            self.model.load_state_dict(unlearnt_dict)

            # Move the model back to the GPU
            self.model.to(self.device)

        # Delete intermediate objects to release memory
        del current_state_dict_cpu
        del unlearnt_dict

    def restore_artm_model(self, previous_state_dict):
        get_accelerator().empty_cache()
        torch.cuda.empty_cache()

        self.model.load_state_dict(previous_state_dict)
        self.model.to(self.device)



    def on_validation_epoch_start(self) -> None:
        get_accelerator().empty_cache()
        torch.cuda.empty_cache()

        if self.edit_started:
            self.previous_state_dict = copy.deepcopy(self.model.state_dict())
            self.artm_edit_before_val(self.model.state_dict())

            get_accelerator().empty_cache()
            torch.cuda.empty_cache()

    # on_validation_epoch_end and another in Neo for validation_epoch_end
    def on_validation_epoch_end(self) -> None:
        if self.edit_started and self.previous_state_dict is not None: 
            self.restore_artm_model(self.previous_state_dict)
            self.previous_state_dict = None 

    # Reduce results from gpus to a single dataframe + determine early stopping
    def validation_epoch_end(self, output):
        if self.hparams.mode in ['unlearn']:

            save_dir = f'llmu_results/{self.hparams.wandb_run_name}'

            if self.init_validation:
                log_col_name = 'init'
            else:
                log_col_name = f'{self.current_epoch:02d}'

            # reduce all output from gpus
            if len(self.hparams.valid_sets) > 1:
                outputs = self.all_gather(output)[self.target_validation_idx]
            else:
                outputs = self.all_gather(output)
            keys = outputs[0].keys()  # [doc_id, loss, acc, el]
            full_output = {k: [] for k in keys}

            # gather all outputs
            for out in outputs:
                for k in keys:
                    full_output[k].append(torch.flatten(out[k]))

            # refactor into pandas favorable format
            for k in keys:
                full_output[k] = torch.cat(full_output[k])
                full_output[k] = torch.flatten(full_output[k]).cpu().numpy()

            self.mem_per_sample = dict(zip(full_output['doc_id'], full_output['acc']))

            if len(full_output['preds'].shape) == 1:
                full_output['preds'] = self.tokenizer.decode(
                    full_output['preds'])
            else:
                full_output['preds'] = self.tokenizer.batch_decode(
                    full_output['preds'])

            plot_accuracy_histogram(full_output['acc'], save_dir, log_col_name)

            # except for 'doc_id' rename all keys to include the epoch
            for k in list(keys):
                full_output[f'{k}_{log_col_name}'] = full_output.pop(k)
            full_output['doc_id'] = full_output.pop(f'doc_id_{log_col_name}')
            df = pd.DataFrame(full_output)

            # append to the df that stores all results from all ddp processes
            df['doc_id'] = df['doc_id'].astype(int)
            df = df.drop_duplicates(['doc_id'])
            df = df.set_index('doc_id')
            self.valid_df = self.valid_df.combine_first(df)
            self.valid_df = self.valid_df.reindex(self.valid_df_index)

            # check early stopping criteria
            ma = df[f'acc_{log_col_name}'].mean()

            if ma < self.hparams.first_threshold and not self.edit_started: 
                logging.info(
                        f'Switched to task arithmetic as memory reached {ma=}. Current value of negative_loss {self.hparams.negative_loss=} (True means GA)')
                self.edit_started = True
                # these can be replaced anyway
                self.hparams.negative_loss = False
               
                self.previous_state_dict = copy.deepcopy(self.model.state_dict())
                self.original_state_dict = {key: value.cpu() for key, value in self.model.state_dict().items()}

            el = df[f'el_{self.el_n_main}-gram_{log_col_name}'].mean()
            if self.current_epoch >= self.hparams.min_train_epochs:
                if ma < self.hparams.ma_threshold: # and el < self.hparams.el_threshold:
                    logging.info(
                        f'Early Stopping as Forgetting Threshold is reached, {ma=}, {el=}')
                    self.trainer.should_stop = True

            file_path = f'{save_dir}/val_data_{log_col_name}.json'
            array_out = convert_numpy_to_list(full_output)
            with open(file_path, 'w') as json_file:
                json.dump(array_out, json_file, indent=2) 

    def validation_step(self, batch, batch_idx, dataloader_idx=-1):
        get_accelerator().empty_cache()
        torch.cuda.empty_cache()

        eval_res = None
        if self.mode == 'general_lm_eval':
            eval_res = self.validation_general_lm(batch)
        elif self.mode == 'unlearn':

            if dataloader_idx in [self.target_validation_idx, -1]:
                eval_res = self.validation_forget(batch)
            else:
                self.validation_general_lm(batch)
        else:
            raise Exception(
                f'Currently not supporting {self.mode}')
        
        return eval_res
