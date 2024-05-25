import logging
import numpy as np

# Convert NumPy arrays to Python lists recursively
def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    else:
        return obj
    
# utils taken from ufl
def normalize_reply(text: str, version=2) -> str:
    """
    Standardize the capitalization and punctuation spacing of the input text.
    Version 1: Fix sentence start casing, and punctuation.
    Version 2: Add trailing period, if missing.
    """

    switch_list = [(' .', '.'), (' ,', ','), (' ?', '?'), (' !', '!'), (" ' ", "'")]

    # add spaces so that words and punctuation can be seaprated
    new_text = text.lower()

    # normalize in case of human:
    for new, old in switch_list:
        new_text = new_text.replace(old, new).replace('  ', ' ')

    # split on punctuation to find sentence boundaries
    # capitalize stuff
    tokens = new_text.split(' ')
    for i in range(len(tokens)):
        if i == 0:
            tokens[i] = uppercase(tokens[i])
        elif tokens[i] in ('i', "i'm", "i've", "i'll", "i'd"):
            tokens[i] = uppercase(tokens[i])
        elif tokens[i] in '?.!' and i < len(tokens) - 1:
            tokens[i + 1] = uppercase(tokens[i + 1])
    new_text = ' '.join(tokens)
    new_text = ' ' + new_text + ' '

    for tup in switch_list:
        new_text = new_text.replace(tup[0], tup[1])

    # get rid of surrounding whitespace
    new_text = new_text.strip()
    new_text = new_text.replace('  ', ' ')

    if version > 1 and new_text and new_text[-1] not in '!.?)"\'':
        new_text += '.'

    return new_text


def uppercase(string: str) -> str:
    """
    Make the first character of the string uppercase, if the string is non-empty.
    """
    if len(string) == 0:
        return string
    else:
        return string[0].upper() + string[1:]


# Init UFL configs that are not given
def default_config(config):
    if 'cache_dir' not in config: 
        config.cache_dir = 'cache'
    if 'seed' not in config:
        config.seed = 42
    if 'privacy_method' not in config:
        config.privacy_method = None
    if 'train_sets' not in config:
        config.train_sets = ""
    if 'valid_sets' not in config:
        config.valid_sets = []
    if 'valid_subset_path' not in config:
        config.valid_subset_path = None
    if 'valid_type_path' not in config:
        config.valid_type_path = None
    if 'learning_rate' not in config:
        config.learning_rate = 5e-5
    if 'negative_loss' not in config:
        config.negative_loss = True
    if 'gradient_accumulation_steps' not in config:
        config.gradient_accumulation_steps = 1
    if 'num_train_epochs' not in config:
        config.num_train_epochs = 0
    if 'num_workers' not in config:
        config.num_workers = 0
    if 'wandb_log' not in config:
        config.wandb_log = False
    if 'strategy' not in config:
        config.strategy = None
    if 'fp16' not in config:
        config.fp16 = False
    if 'check_validation_only' not in config:
        config.check_validation_only = False
    if 'check_val_every_n_epoch' not in config:
        config.check_val_every_n_epoch = 1
    if 'tokenizer' not in config:
        config.tokenizer_name_or_path = config.model_name_or_path
    if 'target_length' not in config:
        config.target_length = None
    if 'el_n' not in config:
        config.el_n = [10]
    if 'el_threshold' not in config:
        config.el_threshold = 0
    if 'ma_threshold' not in config:
        config.ma_threshold = 0
    if 'min_train_epochs' not in config:
        config.min_train_epochs = 0
    if 'do_init_eval' not in config:
        config.do_init_eval = True if config.mode == 'unlearn' else False

def set_console_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s (%(filename)s:%(lineno)d) : %(message)s'
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)