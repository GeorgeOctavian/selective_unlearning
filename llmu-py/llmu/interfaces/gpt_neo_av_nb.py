from llmu.interfaces.gpt_neo_pl import Neo
from llmu.model_edit.arithmetic_edit import TaskVector
# differential privacy decoding with Neo gpt 
import copy
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import os    
import torch

class NeoAVNB(Neo):
    def __init__(self, hparams):
        super(NeoAVNB, self).__init__(hparams)
        self.original_state_dict = copy.deepcopy(self.model.state_dict())

        if hparams.nb_checkpoint is None:
            raise Exception("Unknown checkpoint")
        
        avnb_dir = os.path.join(hparams.cache_dir, hparams.nb_checkpoint)
        _state_dict = get_fp32_state_dict_from_zero_checkpoint(avnb_dir) 
        _state_dict = {k.partition('model.')[2]: _state_dict[k] for k in _state_dict.keys()}
        
        self.neighborhood_state_dict = {}
        for key in _state_dict:
            self.neighborhood_state_dict[key] = _state_dict[key] - self.original_state_dict[key]

        del _state_dict
    

    def training_step(self, batch, batch_idx):
        loss, score = self._step(batch)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        print(f'Training loss is: {loss} @@@@@')
        return loss


    def artm_edit_before_val(self, current_state_dict):
        assert self.hparams.first_coef < 0, "First coefficient has to be negative"
        assert self.hparams.second_coef >= 0, "Second coef must be strictly positive"
        if self.hparams.first_coef is None or self.hparams.second_coef is None:
            raise Exception("None coefficients identified")
        

        with torch.no_grad():
        # Convert current_state_dict to CPU
            current_state_dict_cpu = {key: value.cpu() for key, value in current_state_dict.items()}

            # Compute task drift without unnecessary GPU operations
            unlearnt_dict = {}

            for key in current_state_dict_cpu:

                current_state_dict_cpu[key] = current_state_dict_cpu[key] - self.original_state_dict[key]

                unlearnt_dict[key] = self.original_state_dict[key] + self.hparams.first_coef * current_state_dict_cpu[key] + \
                                    self.hparams.second_coef * self.neighborhood_state_dict[key]

            # Load the new state_dict directly without unnecessary GPU operations
            self.model.load_state_dict(unlearnt_dict)

            # Move the model back to the GPU
            self.model.to(self.device)

        # Delete intermediate objects to release memory
        del current_state_dict_cpu
        del unlearnt_dict


    def restore_artm_model(self, previous_state_dict):
        self.model.load_state_dict(previous_state_dict)
        self.model.to(self.device)


    def on_validation_epoch_start(self) -> None:
        self.previous_state_dict = copy.deepcopy(self.model.state_dict())
        self.artm_edit_before_val(self.model.state_dict())


    def on_validation_epoch_end(self) -> None:
        self.restore_artm_model(self.previous_state_dict)
        self.previous_state_dict = None 


    def validation_step(self, batch, batch_idx, dataloader_idx=-1):

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
