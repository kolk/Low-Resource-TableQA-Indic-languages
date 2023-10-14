from cgi import test
from gc import callbacks
from lib2to3.pgen2 import token
from lib2to3.pgen2.tokenize import tokenize
import os
import transformers
import argparse
import torch
from rouge_score import rouge_scorer, scoring
from datasets import load_from_disk, concatenate_datasets

from transformers import (AutoModelForSeq2SeqLM,
                          AutoConfig,
                          Seq2SeqTrainer,
                          Seq2SeqTrainingArguments,
                          AutoTokenizer,
                          EarlyStoppingCallback,
                          MBartForConditionalGeneration,
                          MBart50TokenizerFast)
from transformers import set_seed
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

os.environ["WANDB_DISABLED"] = "true"
os.environ['TRANSFORMERS_CACHE'] = '~/cache_dir/'
os.environ['HF_DATASETS_CACHE'] = "~/cache_dir/datasets"
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"

parser = argparse.ArgumentParser()
parser.add_argument("--decoder_max_length", default=1024, type=int, help="encoder sequence max length")
parser.add_argument("--pretrained_model_name", type=str, default=None, help="prtrained model name")
parser.add_argument("--language", type=str, default="bn", help="total number of checkpoints to save")
parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--lr_scheduler", default="polynomial", choices=arg_to_scheduler_choices,
                    metavar=arg_to_scheduler_metavar, type=str, help="Learning rate scheduler", )
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for AdamW optimizer.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight Decay for AdamW optimizer.")
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--max_grad_norm", default=0.1, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
parser.add_argument("--use_multiprocessing", default=True, type=bool, help="use multiple processes for data loading")
parser.add_argument("--num_train_epochs", default=30, type=int)
parser.add_argument("--train_batch_size", default=256, type=int)
parser.add_argument("--eval_batch_size", default=256, type=int)
parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
parser.add_argument("--eval_gradient_accumulation", default=1, type=int)
parser.add_argument("--adafactor", action="store_true")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cpu", action="store_true", help="train using cpu")
parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="resume training from a checkpoint")
parser.add_argument("--gradient_checkpointing", action="store_true")
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--save_total_limit", type=int, default=1, help="total number of checkpoints to save")

args = parser.parse_args()
use_cuda = False if args.cpu else True
device = torch.device("cuda" if use_cuda else "cpu")

seed = args.seed

if "facebook/bart" in args.pretrained_model_name:
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
elif "facebook/mbart" in args.pretrained_model_name:
    tokenizer = MBart50TokenizerFast.from_pretrained(args.pretrained_model_name)
config = AutoConfig.from_pretrained(args.pretrained_model_name)

def model_init():
    set_seed(args.seed)
    if "facebook/bart" in args.pretrained_model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_name)
    elif "facebook/mbart" in args.pretrained_model_name:
        model = MBartForConditionalGeneration.from_pretrained(args.pretrained_model_name)
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id["bn_IN"]
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id["bn_IN"]

    model.config.max_length = args.decoder_max_length
    model = model.to(device)
    return model


def get_dataset(lang="bn"):
    if "bart" in config.name_or_path:
        if lang == "en":
            print("Loading Training set tokenized with bart-base tokenizer...")
            train_set = load_from_disk("data/sql2nq_tokenized_train.hf")
            print("Loading Validation set...")
            valid_set = load_from_disk("data/sql2nq_tokenized_dev.hf")
        elif lang == "bn":
            tokenizer.src_lang = "bn_IN"
            tokenizer.tgt_lang = "bn_IN"
            spider_train_set = \
            load_from_disk("data/bengalisql2nq/bengali_spider_tokenized")["train"]
            spider_validation_set = \
            load_from_disk("data/bengalisql2nq/bengali_spider_tokenized")["validation"]
            wikisql_train_set = load_from_disk("data/bengalisql2nq/wikisql_bengali_train_tokenized")
            wikisql_test_set = load_from_disk("data/bengalisql2nq/wikisql_bengali_test_tokenized")
            wikisql_validation_set = load_from_disk("data/bengalisql2nq/wikisql_bengali_valid_tokenized")
            train_set = concatenate_datasets([spider_train_set, wikisql_train_set, wikisql_test_set])
            valid_set = concatenate_datasets([spider_validation_set, wikisql_validation_set])
    return train_set, valid_set, None


# print(f"args.dataset_name {args.dataset_name}")
print(f"Using {args.language}")
train_dataset, valid_dataset, test_dataset = get_dataset(lang="hi")
print("############# Data loading done!#############")


def rouge_metric_builder(tokenizer):
    def compute_rouge_metrics(pred):
        """utility to compute ROUGE during training."""
        # All special tokens are removed.
        pred_ids, labels_ids = pred.predictions, pred.label_ids
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()

        for ref, pred in zip(label_str, pred_str):
            print("target:", ref)
            print("pred:", pred)
            score = scorer.score(ref, pred)
            aggregator.add_scores(score)

        result = aggregator.aggregate()
        return {
            "rouge1": round(result['rouge1'].mid.fmeasure, 4),
            "rouge2": round(result['rouge2'].mid.fmeasure, 4),
            "rougeL": round(result['rougeL'].mid.fmeasure, 4),
        }

    return compute_rouge_metrics

rouge_metric_fn = rouge_metric_builder(tokenizer)

def collate_tokenized(batch):
    """
    Generates tokenized batches
    """
    batch_input_ids, batch_attention_mask, batch_labels = [], [], []
    for sample in batch:
        batch_input_ids.append(torch.tensor(sample['input_ids']))
        batch_attention_mask.append(torch.tensor(sample['attention_mask']))
        batch_labels.append(torch.tensor(sample['labels']))

    return {"input_ids": torch.stack(batch_input_ids).squeeze(),
            "attention_mask": torch.stack(batch_attention_mask).squeeze(),
            "labels": torch.stack(batch_labels).squeeze(),
            }


earlystopping_callback = EarlyStoppingCallback(early_stopping_patience=200)
train_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    no_cuda=args.cpu,
    fp16=True if use_cuda else False,
    save_strategy="epoch",
    save_total_limit=args.save_total_limit,
    logging_steps=100,
    eval_accumulation_steps=args.eval_gradient_accumulation,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    num_train_epochs=args.num_train_epochs,
    seed=seed,
    disable_tqdm=False,
    predict_with_generate=True,
    generation_max_length=args.decoder_max_length,
    generation_num_beams=4,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    dataloader_num_workers=args.num_workers,
    metric_for_best_model="rougeL",
    dataloader_drop_last=True,
    adam_epsilon=args.adam_epsilon,
    weight_decay=args.weight_decay,
    max_grad_norm=args.max_grad_norm,
    lr_scheduler_type=args.lr_scheduler,
    warmup_steps=args.warmup_steps,
    gradient_checkpointing=args.gradient_checkpointing,
    local_rank=args.local_rank,
)

transformers.logging.set_verbosity_info()
trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=collate_tokenized,
    compute_metrics=rouge_metric_fn,
    callbacks=[earlystopping_callback]
)

print("Starting Training...")
if args.resume_from_checkpoint:
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
else:
    trainer.train()
trainer.save_state()
