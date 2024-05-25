import torch
from torch.utils.data import Dataset
import pandas as pd
from llmu.utils.llmu_train_dataset_utils import LLMU_Train_Dataset
from llmu.utils.compute_neighbours import NbLogic

class NbDataset(LLMU_Train_Dataset):

    def __init__(
            self,
            tokenizer,
            dataset_name,
            valid_subset_path,
            type_path,
            input_length,
            output_length,
            device,
            args):
        
        self.device=device
        self.args = args
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.output_length = output_length
        self.dataset_name = dataset_name
        self.type_path = type_path
        self.valid_subset_path = valid_subset_path

        if self.type_path == 'train':
            self.dataset = pd.read_csv(dataset_name, lineterminator='\n')
            batch_size = self.args.train_batch_size * \
                self.args.gradient_accumulation_steps * len(self.args.gpu_list)
            if len(self.dataset) != batch_size:
                raise Exception(
                    "Effective batch size should be the same as length of train set")
        else:
            raise Exception(
                "This class is an interface for training LLMU only")

        self.dataset = self.dataset.dropna()

        # Process the entire dataset here and store it as attributes
        self.prep_for_batch()

    def compute_nb_ds(self):
        nb_logic = NbLogic(self.args, self.device)     
        decoded_sources = []
        for _, data in self.dataset.iterrows():
            # This just puts a limit on the number of tokens in the original dataset
            source, _, _, _, _, _, _ = self.convert_to_features(data)
            source_ids = source["input_ids"].squeeze()

            decoded_sources.append(self.tokenizer.decode(source_ids, skip_special_tokens=True))

        return nb_logic.perturb_and_return(decoded_sources)

    def prep_for_batch(self):
        self.data_dict = {
            "source_ids": [],
            "source_mask": [],
            "target_ids": [],
            "target_mask": [],
            "doc_id": [],
            "task": [],
            "task_type": [],
            "choices": [],
            "answer_index": []
        }
        for data in self.compute_nb_ds():
            try:
                source, targets, doc_id, task, task_type, choices, answer_index = self.cvt_ftr(data)
                source_ids = source["input_ids"].squeeze()
                target_ids = targets["input_ids"].squeeze()
                src_mask = source["attention_mask"].squeeze()
                target_mask = targets["attention_mask"].squeeze()

                # Append data to the corresponding lists in the dictionary
                self.data_dict["source_ids"].append(source_ids)
                self.data_dict["source_mask"].append(src_mask)
                self.data_dict["target_ids"].append(target_ids)
                self.data_dict["target_mask"].append(target_mask)
                self.data_dict["doc_id"].append(doc_id)
                self.data_dict["task"].append(task)
                self.data_dict["task_type"].append(task_type)
                self.data_dict["choices"].append(choices)
                self.data_dict["answer_index"].append(answer_index)
            except Exception as e:
                print(data)

    def cvt_ftr(self, non_batch):

        doc_id = ''

        choices = []
        answer_index = 0
        task, task_type = '', ''
        if self.type_path == 'train':
            input_ = non_batch
            target_ = non_batch
        else:
           raise Exception(
               "The type path should always be train because this is for LLMU training"
           )

        if not task:
            if self.valid_subset_path:
                task = f'{self.dataset_name}_{self.valid_subset_path}'
            else:
                task = f'{self.dataset_name}'

        source = self.tokenizer(
            input_,
            max_length=self.input_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt")

        targets = self.tokenizer(
            target_,
            max_length=self.output_length,
            add_special_tokens=False,
            padding='max_length',
            truncation=True,
            return_tensors="pt")

        # targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length,
        # padding='max_length', truncation=True, return_tensors="pt")
        return source, targets, doc_id, task, task_type, choices, answer_index