from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
import os
import numpy as np
import random 
import datasets
import pandas as pd 
class MIADataset(TorchDataset):
    def __init__(self, 
                 data_dir, 
                 cache_dir,
                 tokenizer,
                 type_path,
                 input_length,
                 output_length,
                 args,
                 key='text',
                 dataset_name='enron',
                 max_length=512):
        
        self.args = args
        self.input_length = input_length
        self.output_length = output_length
        self.type_path = type_path
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.batch_size = self.args.train_batch_size * \
            self.args.gradient_accumulation_steps * len(self.args.gpu_list)
        # preproc_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-small', model_max_length=512, cache_dir=cache_dir)

        if dataset_name == "enron":
            self.final_dataset = self.load_enron_dataset(data_dir, cache_dir, key)[key]
        else: 
            print(f"Loading {dataset_name} dataset")
            if '.csv' in dataset_name:
                self.final_dataset = pd.read_csv(dataset_name, lineterminator='\n')['text']
                self.final_dataset.dropna()
            else:
                data = datasets.load_dataset(dataset_name, split=f'train[:10000]', cache_dir=cache_dir)["document"]
                self.final_dataset = self.process_and_sample(data, None)

        # get the preprocess tokeniser
                # print stats about remainining data
        print(f"Total number of samples: {len(self.final_dataset)}")
        print(f"Average number of words: {np.mean([len(x.split()) for x in self.final_dataset])}")


        if len(self.final_dataset) != self.batch_size:
            raise Exception(
                "Effective batch size should be the same as length of train set")


    def process_and_sample(self, data, preproc_tokenizer):
        
        # strip newlines
        def strip_newlines(text):
            return ' '.join(text.split())
        
        data = list(dict.fromkeys(data))  # deterministic, as opposed to set()


        data = [x.strip() for x in data]

        # remove newlines from each example
        data = [strip_newlines(x) for x in data]

        # try to keep only examples with > 100 words
        #if dataset in ['writing', 'squad', 'xsum']:
        long_data = [x for x in data if len(x.split()) > 100]
        if len(long_data) > 0:
            data = long_data

        
        not_too_long_data = [x for x in data if len(x.split()) < self.max_length]
        if len(not_too_long_data) > 0:
                data = not_too_long_data

        random.shuffle(data)

        data = data[:5_000]

        data = data[:self.batch_size]
        

        return data 
    
       

    def load_enron_dataset(self, data_dir, cache_dir, key):
        cache_path = os.path.join(cache_dir, f"enron_dataset_{self.batch_size}")
        
        # Check if the dataset is already cached
        if os.path.exists(cache_path):
            print(f"Loading Enron dataset from cache: {cache_path}")
            return Dataset.load_from_disk(cache_path)
        
        # If not cached, read emails and create the dataset
        emails = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, 'r', errors='ignore') as f:
                    content = f.read()
                    emails.append(content)

        processed_emails = self.process_and_sample(emails, None)
        
        # Create the dataset
        enron_dataset = Dataset.from_dict({key: processed_emails})
        
        # Save the dataset to the cache directory
        enron_dataset.save_to_disk(cache_path)
        print(f"Enron dataset saved to cache: {cache_path}")
        
        return enron_dataset
    
    def convert_to_features(self, example_batch):
        # this is just 1 email
        input_ = example_batch
        target_ = example_batch

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
        
        return source, targets
        
    def __len__(self):
        return len(self.final_dataset)

    def __getitem__(self, idx):
        email_text = self.final_dataset[idx]
        try:
            source, targets = self.convert_to_features(email_text)
        except:
            print(email_text)

        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids,
                "source_mask": src_mask,
                "target_ids": target_ids,
                "target_mask": target_mask,
                "type_path": self.type_path}