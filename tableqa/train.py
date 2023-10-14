import torch
import transformers
from transformers import (AutoTokenizer,
                          M2M100Tokenizer,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          MBartForConditionalGeneration,
                          M2M100ForConditionalGeneration,
                          EarlyStoppingCallback,
                          set_seed)
from datasets import (load_from_disk,
                      concatenate_datasets,
                      load_metric)
import glob
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_language_model", default=None, type=str,
                    help="checkpoint or huggingface pretrained langauge model path")
parser.add_argument("--decoder_max_length", default=1024, type=int, help="encoder sequence max length")
parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--lr_scheduler",
                    default="polynomial")  # , choices=arg_to_scheduler_choices,  metavar=arg_to_scheduler_metavar, type=str, help="Learning rate scheduler",)
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
parser.add_argument("--train_dataset_path", type=str, default="", help="path to tokenized train dataset in hf format")
parser.add_argument("--validation_dataset_path", type=str, default="", help="path to tokenized validation dataset in hf format")

args = parser.parse_args()
device = torch.device("cpu" if args.cpu else "cuda")

validation_dataset = load_from_disk(args.validation_dataset_path)
train_dataset = load_from_disk(args.tran_dataset_path)
train_dataset = train_dataset.shuffle(seed=42)

print("Training using language model: ", args.pretrained_language_model)
if "mbart-large-50" in args.pretrained_language_model:
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50", src_lang="bn_IN", tgt_lang="bn_IN")
elif "m2m100" in args.pretrained_language_model:
    tokenizer = M2M100Tokenizer.from_pretrained(args.pretrained_language_model, src_lang="bn", tgt_lang="bn")

def model_init():
    set_seed(args.seed)
    if "mbart-large-50" in args.pretrained_language_model:
        model = MBartForConditionalGeneration.from_pretrained(args.pretrained_language_model)
    elif "m2m100" in args.pretrained_language_model:
        model = M2M100ForConditionalGeneration.from_pretrained(args.pretrained_language_model)
    model.config.max_length = args.decoder_max_length
    model = model.to(device)
    return model


def em_metric_builder(tokenizer):
    def compute_em_metrics(pred):
        """utility to compute Exact Match during training."""
        # All special tokens are removed.
        pred_ids, labels_ids = pred.predictions, pred.label_ids
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        em = load_metric("exact_match")
        scores = em.compute(predictions=pred_str, references=label_str, ignore_case=True)
        correct = 0
        for ref, pred in zip(label_str, pred_str):
            print("target:", ref)
            print("pred:", pred)
            correct += (pred == ref)
        accuracy = correct / len(label_str)
        print(f"Exact Match Scores: {scores}")
        return {
            "exact_match": round(scores['exact_match'], 4),
            "accuracy": round(accuracy, 4)
        }

    return compute_em_metrics


em_metric_fn = em_metric_builder(tokenizer)


def collate_tokenized(batch):
    """
    Generates tokenized batches
    """
    batch_input_ids, batch_attention_mask, batch_labels, = [], [], []
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
    # max_steps=15000,
    evaluation_strategy="steps",
    no_cuda=args.cpu,
    fp16=True if not args.cpu else False,
    save_strategy="steps",
    save_total_limit=args.save_total_limit,
    logging_steps=100,
    eval_steps=100,
    save_steps=100,
    eval_accumulation_steps=args.eval_gradient_accumulation,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    num_train_epochs=args.num_train_epochs,
    seed=args.seed,
    disable_tqdm=False,
    predict_with_generate=True,
    generation_max_length=args.decoder_max_length,
    generation_num_beams=4,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    dataloader_num_workers=args.num_workers,
    metric_for_best_model="exact_match",
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
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=collate_tokenized,
    compute_metrics=em_metric_fn,  # rouge_metric_fn,
    callbacks=[earlystopping_callback]
)

print(f"Starting Training from checkpoint {args.resume_from_checkpoint}")
if args.resume_from_checkpoint:
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
else:
    trainer.train()
trainer.save_state()