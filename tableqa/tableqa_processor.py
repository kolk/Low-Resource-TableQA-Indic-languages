import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from datasets import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class TableQAProcessor():
    def __init__(self,
                 training_dataset: Dataset = None,
                 eval_dataset: Dataset = None,
                 test_dataset: Dataset = None,
                 tokenizer: AutoTokenizer = None,
                 max_length: int = 1024,
                 decoder_max_length: int = 1024,
                 is_test: bool = False,
                 **params):
        """
        Generated tokenized batches for training, evaluation and testing of seq2seq task
        """
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.config = AutoConfig.from_pretrained(self.tokenizer.name_or_path)
        self.decoder_max_length = decoder_max_length

        if not is_test:
            self.training_dataset = training_dataset
            self.eval_dataset = eval_dataset
            self.sampler = RandomSampler(data_source=self.training_dataset)
            self.training_generator = DataLoader(self.training_dataset,
                                                 sampler=self.sampler,
                                                 collate_fn=self.collate,
                                                 **params)
            self.eval_generator = DataLoader(self.eval_dataset,
                                             sampler=SequentialSampler(data_source=self.eval_dataset),
                                             collate_fn=self.collate,
                                             **params)
        if is_test:
            self.test_dataset = test_dataset
            self.test_generator = DataLoader(self.test_dataset,
                                             sampler=SequentialSampler(data_source=self.test_dataset),
                                             collate_fn=self.collate_tokenized,
                                             drop_last=True,
                                             **params)

    def collate(self, batch):
        """
        Generates tokenized batches
        """
        tables, table_names, questions, answers = [], [], [], []
        for samp in batch:
            tables.append(samp['tables'])
            table_names.append(samp['table_names'])
            questions.append(samp['question'])
            answers.append(samp['answer'])
        input_encoding = self.tokenizer(tables=tables, table_names=table_names, query=questions, return_tensors="pt",
                                        max_length=self.max_length,
                                        padding=True, truncation='longest_first')  # 'drop_rows_to_fit')

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                answer=answers,
                add_special_tokens=True,
                return_tensors="pt",
                padding=True,
                max_length=self.decoder_max_length,
                truncation='longest_first',
            )

        return {"input_ids": input_encoding["input_ids"],
                "attention_mask": input_encoding["attention_mask"],
                "labels": labels["input_ids"],
                }

    def collate_tokenized(self, batch):
        """
        Generates tokenized batches
        """
        batch_input_ids, batch_attention_mask, batch_labels = [], [], []
        for sample in batch:
            # print("Sample keys", sample.keys())
            batch_input_ids.append(torch.tensor(sample['input_ids']))
            batch_attention_mask.append(torch.tensor(sample['attention_mask']))
            batch_labels.append(torch.tensor(sample['labels']))

        return {"input_ids": torch.stack(batch_input_ids).squeeze(),
                "attention_mask": torch.stack(batch_attention_mask).squeeze(),
                "labels": torch.stack(batch_labels).squeeze(),
                }

