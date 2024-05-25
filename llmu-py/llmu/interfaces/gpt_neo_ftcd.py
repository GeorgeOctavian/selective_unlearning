from llmu.interfaces.gpt_neo_pl import Neo
import torch
from torchmetrics.functional import accuracy
from transformers import LogitsProcessor
import torch.nn as nn
from torch.utils.data import RandomSampler, DataLoader
from llmu.utils.llmu_train_dataset_utils import LLMU_Train_Dataset

"""
This is the constrained decoder class. Check if we can have a metric here that checks MA. 
"""
class Logit_DP_Decoding(LogitsProcessor):
    r"""
    [`LogitsWarper`] and [`LogitsProcessor`] for normalizing the scores using log-softmax. It's important to normalize
    the scores during beam search, after applying the logits processors or warpers, since the search algorithm used in
    this library doesn't do it (it only does it before, but they may need re-normalization) but it still supposes that
    the scores are normalized when comparing the hypotheses.
    """
    def __init__(self, lambda_weight):
        # Lambda weight between [0,1]
        self.lambda_weight = lambda_weight


    def __call__(self, _: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:

        uniform_distribution = torch.ones(scores.shape[0], scores.shape[1]).to('cuda')
        means = torch.mean(scores, dim=-1).unsqueeze(-1)
        uniform_distribution = uniform_distribution * means
        scores = (self.lambda_weight * scores) + (1 - self.lambda_weight) * uniform_distribution

        return scores
    
    
class NeoLLMU(Neo):
    def __init__(self, hparams):
        super(NeoLLMU, self).__init__(hparams)

        self.dp_decoding_logit_processor = Logit_DP_Decoding(lambda_weight=self.hparams.lambda_weight)


    # Training logic for 
    def forward(self, input_ids, attention_mask=None, lm_labels=None):
        
        lm_head_output = self.model(
            input_ids,
            attention_mask=attention_mask
        )

        # These logits will match the ones during generate; so now trick the labels to be different (steer towards a different answer)
        lm_logits = lm_head_output.logits
    
        # https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/models/gpt_neo/modeling_gpt_neo.py#L719
        loss = None
        if lm_labels is not None:
            # move labels to correct device to enable model parallelism
            lm_labels = lm_labels.to(lm_logits.device)
            # Compute loss in fp32 to match with mesh-tf version
            # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., 30:-1, :].contiguous()
            shift_labels = lm_labels[..., 31:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # lm_logits = lm_logits.to(lm_head_output.hidden_states.dtype)
            # loss = loss.to(lm_head_output.hidden_states.dtype)

        return loss, lm_logits

    def _step(self, batch):

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
        loss, score = self._step(batch)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss


    def create_labels_with_dp(self, source_ids):
        input_ids = torch.stack(source_ids).to("cuda")
        # input_ids = train_dataset['source_ids']
        max_len = self.target_length

        # labels, preds = [], []
        preds = []
        for i in range(1, max_len):
            # label = input_ids[..., i]
            prompt = input_ids[..., :i]
            try:
                pred = self.model.generate(prompt, max_length=i + 1, logits_processor=[self.dp_decoding_logit_processor], do_sample=True, top_k=0)[:, -1]
            except IndexError:  # if batch == 1
                pred = self.model.generate(torch.squeeze(
                    prompt), max_length=i + 1).squeeze()[-1]

            # labels.append(torch.squeeze(label))
            preds.append(torch.squeeze(pred))

        first_token_line = input_ids[..., :1].t()
       
        preds.insert(0, torch.squeeze(first_token_line))

        # send back to cpu - might need something smarter here to prevent OOM 
        dp_labels = torch.stack(preds).t().to('cpu')

        # labels = torch.stack(labels)
        same_shape = dp_labels.shape == input_ids.shape
        assert same_shape, 'Inputs and labels have different shapes after dp decoding'

        # get new labels    
        return dp_labels


    def get_training_dataset(self, dataset_name, tokenizer,
                    valid_subset_path, type_path, length=None):
        input_length = length if length else self.hparams.input_length
        output_length = length if length else self.hparams.output_length
        dataset = LLMU_Train_Dataset(
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

        # length = self.target_length

        train_dataset = self.get_training_dataset(
            dataset_name=dataset,
            tokenizer=self.tokenizer,
            valid_subset_path="",
            type_path="train",
            length=length)

        dp_labels = self.create_labels_with_dp(source_ids=train_dataset.data_dict['source_ids'])
        train_dataset.data_dict['target_ids'] = dp_labels

        sampler = RandomSampler(train_dataset)
        dataloader = DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers)
        return dataloader