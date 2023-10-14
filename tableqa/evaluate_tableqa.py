#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from collections import Counter
from transformers import AutoTokenizer
from transformers.models.auto import AutoModelForSeq2SeqLM
from transformers import MBartForConditionalGeneration, M2M100ForConditionalGeneration
from rouge_score import scoring
from collections import defaultdict
import argparse
import torch
from datasets import load_from_disk
from evaluate import load as evaluate_load
import jsonlines
from tableqa_processor import TableQAProcessor

def get_rows_columns_cells(line):
  line=line.lower()
  line=line.split("<কলাম>")[1].strip()
  lines=re.split("\s+row\s+[0-9]+\s+:\s+",line)
  rows=[" | ".join([cell.strip() for cell in row.split("|")]) for row in lines[1:]]
  cells=[cell.strip() for row in lines[1:] for cell in row.split("|")]
  columns=[" | ".join([elem.strip() for elem in elems]) for elems in list(zip(*[row.split(" | ") for row in lines]))]
  return rows,columns,cells

def get_rows_columns_cells_bangla(line):
    try:
      line=line[line.index(" ")+1:]
      lines=line.split("<রো ")
      lines=[item[item.index(" ")+1:] if i>0 and " " in item  else item for i,item in enumerate(lines)]
      lines=[line for line in lines if line]
      rows=[" | ".join([cell.strip() for cell in row.split("|")]) for row in lines[1:]]
      cells=[cell.strip() for row in lines[1:] for cell in row.split("|")]
      columns=[" | ".join([elem.strip() for elem in elems]) for elems in list(zip(*[row.split(" | ") for row in lines]))]
    except:
       return [], [], []
    return rows,columns,cells


def get_correct_total_prediction(target_str,pred_str):
  target_rows,target_columns,target_cells=get_rows_columns_cells_bangla(target_str)
  prediction_rows,prediction_columns,prediction_cells=get_rows_columns_cells_bangla(pred_str)
  common_rows = Counter(target_rows) & Counter(prediction_rows)
  common_rows = list(common_rows.elements())
  common_columns = Counter(target_columns) & Counter(prediction_columns)
  common_columns = list(common_columns.elements())
  common_cells = Counter(target_cells) & Counter(prediction_cells)
  common_cells = list(common_cells.elements())
  return {"target_rows":target_rows,
          "target_columns":target_columns,
          "target_cells":target_cells,
          "pred_rows":prediction_rows,
          "pred_columns":prediction_columns,
          "pred_cells":prediction_cells,
          "correct_rows":common_rows,
          "correct_columns":common_columns,
          "correct_cells":common_cells}


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=1, type=int, help="inference batch size")
parser.add_argument("--cpu", action='store_true', help="Load the model in cpu")
parser.add_argument("--pretrained_model_name", default=None, type=str, help="huggingface pretrained language model name or local path to language model")
parser.add_argument("--generation_max_length", type=int, default=1024, help="max generation sequence length")
parser.add_argument("--validation_dataset_path", type=str, default=None, help="path to validation dataset in huggingface Dataset format")
parser.add_argument("--predictions_save_path", type=str, help="path for predictions to be saved in")

args = parser.parse_args()
print(args)
validation_dataset = load_from_disk(args.validation_dataset_path)
if "mbart" in args.pretrained_model_name:
    model = MBartForConditionalGeneration.from_pretrained(args.pretrained_model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, src_lang="bn_IN", tgt_lang="bn_IN")
    forced_bos_id = forced_bos_token_id = tokenizer.lang_code_to_id["bn_IN"]
elif "m2m" in args.pretrained_model_name:
    model = M2M100ForConditionalGeneration.from_pretrained(args.pretrained_model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, src_lang="bn", tgt_lang="bn")
    forced_bos_id = tokenizer.get_lang_id("bn")
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_name)
device = torch.device("cuda") if not args.cpu else torch.device("cpu")
model.to(device).eval()
outputs = defaultdict(list)

print(f"Evaluating on {len(validation_dataset)} samples")
test_processor = TableQAProcessor(test_dataset=validation_dataset,
                                  batch_size=args.batch_size,
                                  decoder_max_length=args.generation_max_length,
                                  tokenizer=tokenizer,
                                  is_test=True)

suffix = args.pretrained_model_name[args.pretrained_model_name.rfind("/")+1:]
total_columns_in_dataset = 0
total_rows_in_dataset = 0
total_cells_in_dataset = 0
total_correct_rows = 0
total_correct_columns = 0
total_correct_cells = 0
total_prediced_rows_in_dataset = 0
total_predicted_columns_in_dataset = 0
total_predicted_cells_in_dataset = 0

exact_match_metric = evaluate_load("exact_match")
aggregator_em = scoring.BootstrapAggregator()
print('Starting Inference')
tensor_device = "cpu" if args.cpu else "cuda"
predictions, references =[],[]


with jsonlines.open(f"{args.predictions_save_path}predictions_{suffix}.jsonlines", "w", flush=True) as f_pred:
    for i, batch in enumerate(test_processor.test_generator):
        batch_sz = len(batch["input_ids"])
        question = [tokenizer.decode(samp.to(tensor_device), skip_special_tokens=True, clean_up_tokenizatimodon_spaces=False) for
                    samp in batch["input_ids"]]
        prediction = model.generate(batch["input_ids"].to(tensor_device), num_beams=5, return_dict_in_generate=True,
                                    output_scores=True, max_length=args.generation_max_length, forced_bos_token_id=forced_bos_id)
        seq_len = prediction["sequences"].shape[1]
        answer = [tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True) for pred in
                  prediction['sequences']]
        target = [tokenizer.decode(samp, skip_special_tokens=True, clean_up_tokenization_spaces=False) for samp in
                  batch["labels"]]
        assert len(target) == len(answer)
        predictions.extend(prediction)
        references.extend(target)
        em_results = exact_match_metric.compute(predictions=answer, references=target)
        for ques, tgt, ans in zip(question, target, answer):
            em_score = exact_match_metric.compute(predictions=[ans.strip()], references=[tgt.strip()])
            aggregator_em.add_scores(em_score)

            print("question:", ques)
            print("target:", tgt)
            print("prediction:", ans)
            ans = ans.replace("\n\r", " ").replace("\n", " ")
            f_pred.write({"prediction": ans.lower().strip(), "target":tgt.strip().lower()})
            print()

            statistics = get_correct_total_prediction(tgt.strip().lower(), ans.strip().lower())
            total_columns_in_dataset += len(statistics['target_columns'])
            total_rows_in_dataset += len(statistics['target_rows'])
            total_cells_in_dataset += len(statistics['target_cells'])
            total_correct_columns += len(statistics['correct_columns'])
            total_correct_rows += len(statistics['correct_rows'])
            total_correct_cells += len(statistics['correct_cells'])
            total_prediced_rows_in_dataset += len(statistics['pred_rows'])
            total_predicted_columns_in_dataset += len(statistics['pred_columns'])
            total_predicted_cells_in_dataset += len(statistics['pred_cells'])

em_result = aggregator_em.aggregate()
print(f"exact_match: {round(em_result['exact_match'].mid,4)}")

row_accuracy = total_correct_rows / total_rows_in_dataset
column_accuracy = total_correct_columns / total_columns_in_dataset
cell_accuracy = total_correct_cells / total_cells_in_dataset
print(f"row accuracy {round(row_accuracy, 4)}")
print(f"column accuracy {round(column_accuracy, 4)}")
print(f"cell accuracy {round(cell_accuracy, 4)}")
print()
row_precision = total_correct_rows / total_prediced_rows_in_dataset
row_recall = total_correct_rows / total_rows_in_dataset
print(f"row precision {row_precision}")
print(f"row recall {row_recall}")
print(f"row F1 {(2*row_precision*row_recall)/(row_precision+row_recall)}")
print()
column_precision = total_correct_columns / total_predicted_columns_in_dataset
column_recall = total_correct_columns / total_columns_in_dataset
print(f"column_precision {column_precision}")
print(f"column_recall {column_recall}")
print(f"column F1 {(2*column_precision*column_recall)/(column_precision+column_recall)}")
print()
cell_precision = total_correct_cells / total_predicted_cells_in_dataset
cell_recall = total_correct_cells / total_cells_in_dataset
print(f"cell_precision {cell_precision}")
print(f"cell_recall {cell_recall}")
print(f"cell F1 {(2*cell_precision*cell_recall)/(cell_precision+cell_recall)}")