# LowResourceTableQA
**Datasets**
  - Download synthetic BanglaTabQA data (training+validation) at [BanglaTabQA dataset](https://surfdrive.surf.nl/files/index.php/s/slYoi2DZK5ehu0u)
  - Download manually annotated BanglaTabQA [test](data/banglaTabQA_test_set.jsonl) data
  - Download [HindiTabQA dataset](https://surfdrive.surf.nl/files/index.php/s/t49Q7q7pwC35lFj)
  - Download manually annotated HindiTabQA [test](data/hindiTabQA_test_set.jsonl) data

**Model Checkpoints**
  - **BanglaTabQA Models**
     -  Download [`BnTQA-mBart`](https://huggingface.co/vaishali/BnTQA-mBart) 
     -  Download [`BnTQA-M2M`](https://huggingface.co/vaishali/BnTQA-M2M) 
     -  Download [`BnTQA-llama`](https://huggingface.co/vaishali/BnTQA-Llama) 
 - **HindiTabQA Models**  
    - Download [`HiTQA-mBart`](https://huggingface.co/vaishali/HiTQA-mBart)
    - Download [`HiTQA-M2M`](https://huggingface.co/vaishali/HiTQA-M2M)
   - Download [`HiTQA-BnTQA`](https://huggingface.co/vaishali/HiTQA-BnTQA)
   - Download [`HiTQA-llama`](https://huggingface.co/vaishali/HiTQA-Llama)

**Loading BanglaTabQA Model Checkpoints**

  - *BnTQA-mBart* 
```
from transformers import MBartForConditionalGeneration
model = MBartForConditionalGeneration.from_pretrained("vaishali/BnTQA-mBart")
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, src_lang="bn_IN", tgt_lang="bn_IN")
forced_bos_id = forced_bos_token_id = tokenizer.lang_code_to_id["bn_IN"]
```
- *BnTQA-M2M*
```
from transformers import M2M100ForConditionalGeneration
model = M2M100ForConditionalGeneration.from_pretrained("vaishali/BnTQA-M2M")
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, src_lang="bn", tgt_lang="bn")
forced_bos_id = tokenizer.get_lang_id("bn")
```
- *BnTQA-Llama*
```
from transformers import M2M100ForConditionalGeneration
model = M2M100ForConditionalGeneration.from_pretrained("vaishali/BnTQA-Llama")
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, src_lang="bn", tgt_lang="bn")
forced_bos_id = tokenizer.get_lang_id("bn")
```

**Loading HindiTabQA Model Checkpoints**

  - *HiTQA-mBart or HiTQA-BnTQA*
```
from transformers import MBartForConditionalGeneration
model_hitqa_mbart = MBartForConditionalGeneration.from_pretrained("vaishali/HiTQA-mBart")
model_hitqa_bntqa = MBartForConditionalGeneration.from_pretrained("vaishali/HiTQA-BnTQA")
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, src_lang="hi_IN", tgt_lang="hi_IN")
forced_bos_id = forced_bos_token_id = tokenizer.lang_code_to_id["hi_IN"]
```
- *HiTQA-M2M*
```
from transformers import M2M100ForConditionalGeneration
model = M2M100ForConditionalGeneration.from_pretrained("vaishali/HiTQA-M2M")
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, src_lang="hi", tgt_lang="hi")
forced_bos_id = tokenizer.get_lang_id("hi")
```
- *HiTQA-Llama*
```
from transformers import M2M100ForConditionalGeneration
model = M2M100ForConditionalGeneration.from_pretrained(args.pretrained_model_name)
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, src_lang="bn", tgt_lang="bn")
forced_bos_id = tokenizer.get_lang_id("bn")
```

**Bengali SQL query creation**
```
python data_generation/extract_wikitables.py --table_language "bn" --data_save_path "data/bengali_tables.jsonl" --max_table_cells 500
python data_generation/create_sql_samples.py --table_language "bn" --data_save_path "data/bengali_tables.jsonl" --max_table_cells 500
python data_generation/process_code_mixed_sql.py --input_file "data/bengali_sql/non_numeric_code_mixed.jsonl"  \
                                                 --output_file "data/bengali_sql/non_numeric_full_indic.jsonl" \
                                                 --table_language "bn" --sql_language "bengali"
```

**Data Generation Training Process: SQL2NQ**

```
python train_sql2nq.py --pretrained_model_name "facebook/mbart-large-50" \
                --learning_rate 1e-4 --train_batch_size 4 --eval_batch_size 4 --gradient_accumulation_steps 64 --num_train_epochs 60 \
                --use_multiprocessing False --num_workers 2 --decoder_max_length 1024  --language "bn" \
                --output_dir "models/sql2nq"  --seed 45 \
                --save_total_limit 1  
```

**Training Process: Low-Resource Table QA**

Arguments for Bengali TableQA training:
```

python tableqa/train.py --pretrained_language_model "facebook/mbart-large-50" --learning_rate 1e-4 \
                --train_batch_size 2 --eval_batch_size 2 --gradient_accumulation_steps 64 --num_train_epochs 8 \
                --use_multiprocessing False --num_workers 2 --decoder_max_length 1024 \
                 --seed 42 --decoder_max_length 1024 --language "bn" \
                --output_dir "experiments/banglaTabQA_mbart" 

```

Arguments for Hindi TableQA training:
```
python tableqa/train.py --pretrained_language_model "facebook/mbart-large-50" --learning_rate 1e-4 \
                --train_batch_size 2 --eval_batch_size 2 --gradient_accumulation_steps 64 --num_train_epochs 8 \
                --use_multiprocessing False --num_workers 2 --decoder_max_length 1024  \
                 --seed 42 --decoder_max_length 1024 --language "hi" \
                --output_dir "experiments/hindiTabQA_mbart" 
```

Arguments for Bengali Table QA evaluation:
```
python tableqa/evaluate_tableqa.py --pretrained_model_name "vaishali/BnTQA-mBart" \
                --batch_size 2 --generation_max_length 1024 \
                --validation_dataset_path "data/mbart-50_tokenized/mbart-50_validation.hf" \
                --predictions_save_path "experiments/predictions/mbart-50_validation.jsonl" 
```
