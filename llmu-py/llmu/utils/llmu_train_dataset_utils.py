import torch
from torch.utils.data import Dataset
import pandas as pd

# process data in the beginning to ensure generation process for targets is static
class LLMU_Train_Dataset(Dataset):
    def __init__(
            self,
            tokenizer,
            dataset_name,
            valid_subset_path,
            type_path,
            input_length,
            output_length,
            args):
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
        
        for index, data in self.dataset.iterrows():
            try:
                source, targets, doc_id, task, task_type, choices, answer_index = self.convert_to_features(data)
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

    def __len__(self):
        return len(self.data_dict["source_ids"])

    def convert_to_features(self, example_batch):
        try:
            doc_id = torch.tensor(example_batch['doc_id'], dtype=torch.int)
        except KeyError:
            doc_id = ''

        choices = []
        answer_index = 0
        task, task_type = '', ''
        if self.type_path == 'train':
            input_ = example_batch['text']
            target_ = example_batch['text']
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

    def __getitem__(self, index):
        return {
            "source_ids": self.data_dict["source_ids"][index],
            "source_mask": self.data_dict["source_mask"][index],
            "target_ids": self.data_dict["target_ids"][index],
            "target_mask": self.data_dict["target_mask"][index],
            "doc_id": self.data_dict["doc_id"][index],
            "task": self.data_dict["task"][index],
            "task_type": self.data_dict["task_type"][index],
            "choices": self.data_dict["choices"][index],
            "answer_index": self.data_dict["answer_index"][index]
        }