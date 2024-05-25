from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from llmu.utils.ufl_utils import default_config, set_console_logger
from llmu.utils.logging_utils import MetricTracker
from llmu.interfaces.generic_training_pip import GenericAutoregressiveModule
from llmu.interfaces.gpt_neo_pl import Neo
from llmu.interfaces.gpt_neo_dp import NeoDP
from llmu.interfaces.gpt_neo_ftcd import NeoLLMU
from llmu.interfaces.gpt_neo_mia import NeoMIA
from llmu.interfaces.gpt_neo_av import NeoAV
from llmu.interfaces.gpt_neo_av_nb import NeoAVNB
from llmu.interfaces.gpt_neo_av_ifnb import NeoAVIFNB
from llmu.interfaces.gpt_neo_av_peft import NeoAVP
import torch
import os
import argparse
from argparse import ArgumentParser
import json 

from pytorch_lightning.loggers import WandbLogger

if __name__ == '__main__':
    # added this based on PL warning
    torch.set_float32_matmul_precision('medium')

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Parsing Arguments
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    arg_ = parser.parse_args()
    if arg_.config is None:
        raise NameError("Include a config file in the argument please.")

    # Getting configurations
    config_path = arg_.config
    with open(config_path) as config_file:
        config = json.load(config_file)
    config = argparse.Namespace(**config)
    # initialise config with some defaults
    default_config(config)

    # Set console logger
    set_console_logger()

    seed_everything(config.seed, workers=True)
    cache_dir = config.cache_dir
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    # change this to something more interesting
    if not os.path.exists(config.results_dir):
        os.makedirs(config.results_dir)

    # Use string manipulation to extract the last component after the last "/"
    if hasattr(config, 'train_set'): 
        train_ds_name = config.train_set.split("/")[-1]
    else:
        train_ds_name = "None"
    model_name = config.model_name_or_path.split("/")[-1]
    config.wandb_run_name = f'{config.wandb_run_name}_{config.mode}_{config.privacy_method}_{train_ds_name}_{model_name}_seed_{config.seed}_bs_{config.eval_batch_size}_lr_{config.learning_rate}'
    try:
        config.wandb_run_name = f'{config.wandb_run_name}_{config.mode}_{config.privacy_method}_{train_ds_name}_{model_name}_seed_{config.seed}_coefs_{config.first_coef}_{config.second_coef}_epc_{config.nb_epoch}_bs_{config.eval_batch_size}_lr_{config.learning_rate}'
    except AttributeError:
        print("Revert to default wandb run name !")
    
    save_dir = f'{config.results_dir}/{config.wandb_run_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # Set wandb logger
    if config.wandb_log:
        wandb_logger = WandbLogger(
            project=config.wandb_project,
            name=config.wandb_run_name)
    else:
        wandb_logger = None



   
    if "airnb" in config.privacy_method:
        checkpoint_dir = f'{train_ds_name}_{model_name}_{config.mask_model}_num_perturb_{config.n_perturbations}_pct_{config.mask_pct}'
        config.nb_checkpoint = checkpoint_dir + f"/model-checkpoint-epoch={config.nb_epoch}.ckpt"
        if not os.path.exists(os.path.join(cache_dir, config.nb_checkpoint)):
            checkpoint_callback = ModelCheckpoint(dirpath=cache_dir + f'/{checkpoint_dir}',
                                                filename='model-checkpoint-{epoch:02d}',
                                                every_n_epochs=1,
                                                save_top_k=-1
                                                )
            callbacks = [checkpoint_callback]
            train_params = dict(
  
                accumulate_grad_batches=config.gradient_accumulation_steps,
                accelerator='gpu',
                
                devices=config.gpu_list,
   
                max_epochs=int(config.num_train_epochs_nb),
                precision='16' if config.fp16 else 32,

                callbacks=callbacks,
  
                strategy=config.strategy,
                num_sanity_val_steps=0,
                limit_val_batches=0,
                log_every_n_steps=1
            )

            trainer = Trainer(**train_params)

            # there's just one model for now; this may change depending on what we want to fine-tune for
            model = GenericAutoregressiveModule(config)
            trainer.fit(model)

            del model 
            del trainer

            # Free CUDA memory
            torch.cuda.empty_cache()

    # metric_tracker_callback = MetricTracker(config.wandb_run_name)
    callbacks = [MetricTracker(config.wandb_run_name)]
    # Setting for pytorch lightning trainer
    train_params = dict(

        accumulate_grad_batches=config.gradient_accumulation_steps,
        accelerator='gpu',
        # NOTE: devices: The devices to use. Can be set to a positive number (int or str), a sequence of device indices
        devices=config.gpu_list,
        max_epochs=int(config.num_train_epochs),
        precision='16' if config.fp16 else 32,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        enable_checkpointing=False,
        callbacks=callbacks,
        logger=wandb_logger,
        strategy=config.strategy,
        num_sanity_val_steps=0,
        limit_val_batches=1,
        log_every_n_steps=1
    )

    trainer = Trainer(**train_params)

    print(torch.version.cuda)
    
    if config.privacy_method == "llmu":
        model = NeoLLMU(config)
    elif config.privacy_method == "dp":
        model = NeoDP(config)
    elif config.privacy_method == "ufl":
        model = Neo(config)
    elif config.privacy_method == "ufl_mia":
        model = NeoMIA(config)
    elif config.privacy_method == "air":
        model = NeoAV(config)
    elif config.privacy_method == "air_peft":
        model = NeoAVP(config)
    elif config.privacy_method == "airnb":
        model = NeoAVNB(config)
    elif config.privacy_method =="airnb_if":
        model = NeoAVIFNB(config)
    else:
        raise Exception(
            f'Unknown privacy method')

    if config.privacy_method == "dp" or config.check_validation_only:
        trainer.validate(model)
    else:
        if config.do_init_eval:
            trainer.validate(model)
        trainer.fit(model)





