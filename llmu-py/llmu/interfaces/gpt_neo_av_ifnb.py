from llmu.interfaces.gpt_neo_pl import Neo
from torchmetrics.functional import accuracy
# differential privacy decoding with Neo gpt 
import copy
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import os    
import torch
from transformers import LogitsProcessor

class Logit_Forget_Decoder(LogitsProcessor):
    r"""
    [`LogitsWarper`] and [`LogitsProcessor`] for normalizing the scores using log-softmax. It's important to normalize
    the scores during beam search, after applying the logits processors or warpers, since the search algorithm used in
    this library doesn't do it (it only does it before, but they may need re-normalization) but it still supposes that
    the scores are normalized when comparing the hypotheses.
    """
    def __init__(self, og_state_dict, cache_dir, nb_checkpoint, model, first_coef=-1, second_coef=1):

        avnb_dir = os.path.join(cache_dir, nb_checkpoint)
        _state_dict = get_fp32_state_dict_from_zero_checkpoint(avnb_dir) 
        _state_dict = {k.partition('model.')[2]: _state_dict[k] for k in _state_dict.keys()}

        self.neighborhood_state_dict = _state_dict  
        self.original_state_dict = og_state_dict

        self.first_coef = first_coef
        self.second_coef = second_coef

        self.model = model


    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        

        cur_state_dict = copy.deepcopy(self.model.state_dict())

        # step 1: load the model weights in self model from original and compute scores
        self.model.load_state_dict(self.original_state_dict)
        output_og = self.model(input_ids = input_ids)
        scores_og = torch.squeeze(output_og['logits'][:,-1,:], dim=1)

        # step 2: load the model weights from checkpoint with neighbourhood/retain and compute scores 
        self.model.load_state_dict(self.neighborhood_state_dict)
        output_nb = self.model(input_ids = input_ids)
        scores_nb = torch.squeeze(output_nb['logits'][:,-1,:], dim=1)

        # step 3: reload the original weights in here, then continue :) 
        self.model.load_state_dict(cur_state_dict)
        
        scores = scores_og + self.first_coef * (scores - scores_og) +  self.second_coef * (scores_nb - scores_og)

        return scores
    

class NeoAVIFNB(Neo):
    def __init__(self, hparams):
        super(NeoAVIFNB, self).__init__(hparams)
        self.original_state_dict = copy.deepcopy(self.model.state_dict())

        if hparams.nb_checkpoint is None:
            raise Exception("Unknown checkpoint")
        
        self.decoder_hook = Logit_Forget_Decoder(self.original_state_dict, hparams.cache_dir,
                                                 hparams.nb_checkpoint, self.model, hparams.first_coef, 
                                                 hparams.second_coef)


    def training_step(self, batch, batch_idx):
        loss, score = self._step(batch)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        print(f'Training loss is: {loss} @@@@@')
        return loss


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
    
    def validation_ma(self, batch, dataset_name):
        input_ids = batch['source_ids']
        max_len = self.target_length

        labels, preds = [], []
        for i in range(1, max_len):
            label = input_ids[..., i]
            prompt = input_ids[..., :i]
            try:
                pred = self.model.generate(prompt, max_length=i + 1, logits_processor=[self.decoder_hook], do_sample=True)[:, -1]
            except IndexError:  # if batch == 1
                pred = self.model.generate(torch.squeeze(
                    prompt), max_length=i + 1).squeeze()[-1]

            labels.append(torch.squeeze(label))
            preds.append(torch.squeeze(pred))

        preds = torch.stack(preds)
        labels = torch.stack(labels)

        score = accuracy(preds, labels, ignore_index=-100)
        self.log(
            f'{dataset_name}/acc',
            score,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True)

        # return individual example results for logging
        return torch.t(preds), torch.t(labels)

    def validation_el(self, batch, dataset_name):

        input_ids = batch['source_ids']
        max_len = self.target_length

        batch_size = input_ids.shape[0]
        N = self.el_n
        numerator = {n: [0] * batch_size for n in N}

        for i in reversed(range(1, max_len)):
            label = input_ids[..., i:max_len]
            prompt = input_ids[..., :i]

            pred = self.model.generate(prompt, max_length=max_len, logits_processor=[self.decoder_hook], do_sample=True)[..., i:]

            for example_idx in range(batch_size):
                p, l = pred[example_idx], label[example_idx]
                # extraction likelihood
                for n in N:
                    p_ngram = self.ngram_of_1D_tensor(p, n)
                    l_ngram = self.ngram_of_1D_tensor(l, n)
                    l_unique = set(l_ngram)
                    p_tp = [i for i in p_ngram if i in l_unique]
                    try:
                        p_acc = len(p_tp) / len(l_ngram)
                        numerator[n][example_idx] += p_acc
                    except ZeroDivisionError:  # n-gram isn't defined
                        pass

        el_score = {n: [0] * batch_size for n in N}
        for n in N:
            for i, _ in enumerate(numerator[n]):
                el_score[n][i] = numerator[n][i] / \
                    (max_len - 1 - (n - 1))

        for n in N:
            self.log(f'{dataset_name}/el_{n}-gram',
                     sum(el_score[n]) / len(el_score[n]),
                     prog_bar=True,
                     logger=True,
                     add_dataloader_idx=False,
                     sync_dist=True)

        # return individual example results for logging
        ret = {}
        for k in el_score.keys():
            ret[f'el_{k}-gram'] = torch.Tensor(el_score[k])
        return ret
    
    def dialog_evaluation(self, padding_length, task, batch):
        source_ids = batch["source_ids"].tolist()
        target_ids = batch["target_ids"].tolist()
        batch_size = len(source_ids)

        inps, cont_toks_list, inplens = [], [], []
        for i in range(batch_size):
            context_enc = self.get_rid_of_pad(source_ids[i])
            continuation_enc = self.get_rid_of_pad(target_ids[i])

            # sanity check
            assert len(context_enc) > 0
            assert len(continuation_enc) > 0
            assert len(continuation_enc) <= self.max_length

            inp = torch.tensor(
                (context_enc + continuation_enc)[-(padding_length):],
                dtype=torch.long
            ).to(self.device)
            inplen, = inp.shape
            cont = continuation_enc

            # pad length from seq to padding_length
            inp = torch.cat([
                inp,  # [seq]
                # [padding_length - seq]
                torch.zeros(padding_length - inplen,
                            dtype=torch.long).to(inp.device) + self.tokenizer.pad_token_id
            ], dim=0)
            inps.append(inp.unsqueeze(0))  # [1, padding_length]
            cont_toks_list.append(cont)
            inplens.append(inplen)

        batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length
        multi_logits = self._model_call(batched_inps)  # [batch, padding_length, vocab]
        
        full_logits, full_cont_toks = [], []
        for logits, inp, inplen, cont_toks \
                in zip(multi_logits, inps, inplens, cont_toks_list):

            # Slice to original seq length
            contlen = len(cont_toks)

            if contlen >= padding_length:
                cont_toks = cont_toks[:int(padding_length / 2)]
                contlen = len(cont_toks)

            # [seq, vocab]
            logits = logits[inplen - contlen - 1:inplen - 1]
            # Check if per-token argmax is exactly equal to continuation
            cont_toks = torch.tensor(
                cont_toks, dtype=torch.long).to(self.device)  # [seq]

            assert logits.shape[0] == cont_toks.shape[0]

            full_logits.append(logits)
            full_cont_toks.append(cont_toks)

        full_logits = torch.cat(full_logits)
        full_cont_toks = torch.cat(full_cont_toks)

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(full_logits, full_cont_toks)
        
        generate_input = []
        for source_id in source_ids:
            inplen = len(source_id)
            inp = torch.tensor(source_id, dtype=torch.long).to(self.device)
            inp = torch.cat([
                torch.zeros(padding_length - inplen,
                            dtype=torch.long).to(inp.device) + self.tokenizer.pad_token_id,
                inp
            ], dim=0)
            generate_input.append(inp.unsqueeze(0))  # [1, padding_length]

        inputs = torch.cat(generate_input, dim=0)
        attention_masks = inputs.ne(self.tokenizer.pad_token_id).long()
        #generated_ids = self.model.generate(inputs, attention_mask=attention_masks, max_new_tokens=32)[:, padding_length:]
        generated_ids = self.model.generate(inputs, attention_mask=attention_masks, max_new_tokens=32, do_sample=True, logits_processor=[self.decoder_hook])[:, padding_length:]
        generated_text = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        generated_text = [t.split('\nUser ')[0] for t in generated_text]
        target_text = self.tokenizer.batch_decode(target_ids, skip_special_tokens=True)

        # Debugging
        source_text = self.tokenizer.batch_decode(source_ids, skip_special_tokens=True)
        for s, g, t in zip(source_text, generated_text, target_text):
            print('---------------------')
            print(f'[Prefix] {s}')
            print(f'[Ground Truth] {t}')
            print(f'[Generated] {g}')
            print('---------------------')
        
        f1_batched = 0
        for g, t in zip(generated_text, target_text):
            f1_batched += self._f1_score(g, t)

        unigram_f1 = f1_batched / batch_size

        self.log(
            f'{task}/loss',
            loss,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True),
        self.log(
            f'{task}/f1',
            unigram_f1,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True)
