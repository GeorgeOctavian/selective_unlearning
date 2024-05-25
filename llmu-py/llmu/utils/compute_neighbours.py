import re 
import transformers
import numpy as np 
import tqdm
import functools
import time
import torch
class NbLogic:
    def __init__(self, hparams, device):
        self.hparams = hparams
        self.device = device
        
        self.pattern = re.compile(r"<extra_id_\d+>")

        print(f'Loading mask filling model {self.hparams.mask_model}...')

        n_positions = 512
        
        self.mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.hparams.mask_model, 
                                                                            #  **int8_kwargs, **half_kwargs, 

                                                                             cache_dir=self.hparams.cache_dir)
        self.mask_tokenizer = transformers.AutoTokenizer.from_pretrained(self.hparams.mask_model, 
                                                                         model_max_length=n_positions, 
                                                                         cache_dir=self.hparams.cache_dir)
        try: 
            self.mask_model.cpu()
        except NameError:
            print("moving mask model to CPU failed due to name error")
            pass 


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
    
    def perturb_and_return(self, decoded_batch):

        self.load_mask_model()

        if self.hparams.mask_pct is None:
            raise Exception("masking percentage is not defined in the cofniguration")
        perturb_function = functools.partial(self.perturb_texts, span_length=2, pct=self.hparams.mask_pct, ceil_pct=False)
        
        perturbed_decoded = perturb_function([x for x in decoded_batch for _ in range(self.hparams.n_perturbations)])

        self.release_mask_model()

        del self.mask_model
        torch.cuda.empty_cache()

        return perturbed_decoded

    
    def load_mask_model(self):
        print('MOVING MASK MODEL TO GPU...', end='', flush=True)
        start = time.time()

        # if not args.random_fills:
        self.mask_model.to(self.device)
        print(f'DONE ({time.time() - start:.2f}s)')

    def release_mask_model(self):
        print('MOVING MASK BACK TO CPU...', end='', flush=True)
        start = time.time()
        try:
            self.mask_model.cpu()
        except NameError:
            print("failed moving mask model to cpu")
            pass
        print(f'DONE ({time.time() - start:.2f}s)')

