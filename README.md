# Table Question Answering for Low-resourced Indic Languages

**Data Creation Process**
![Data Creation Pipeline Diagram](images/data_creation.png)

**Datasets**
  - Download synthetic BanglaTabQA data (training+validation) at [BanglaTabQA dataset](https://surfdrive.surf.nl/files/index.php/s/slYoi2DZK5ehu0u)
  - Download manually annotated BanglaTabQA [test](data/banglaTabQA_test_set.jsonl) data
  - Download [HindiTabQA dataset](https://surfdrive.surf.nl/files/index.php/s/t49Q7q7pwC35lFj)
  - Download manually annotated HindiTabQA [test](data/hindiTabQA_test_set.jsonl) data
    
Alternatively, load the BanglaTabQA dataset from huggingface hub:
```
from datasets import load_dataset
banglatabqa = load_dataset("vaishali/banglaTabQA")
training_set, validation_set, test_set = banglatabqa['training'], banglatabqa['validation'], banglatabqa['test']
```

Alternatively, load the HindiTabQA dataset from  huggingface hub:
```
from datasets import load_dataset
hinditabqa = load_dataset("vaishali/hindiTabQA")
training_set, validation_set, test_set = hinditabqa['training'], hinditabqa['validation'], hinditabqa['test']
```
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
tokenizer = AutoTokenizer.from_pretrained("vaishali/BnTQA-M2M", src_lang="bn", tgt_lang="bn")
forced_bos_id = tokenizer.get_lang_id("bn")
```
- *BnTQA-Llama*
```
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
model_name = LLAMA-2-7b_DIRECTORY
adapters_name = "vaishali/BnTQA-Llama"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map={"": 0}
    )
    model = PeftModel.from_pretrained(model, adapters_name)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
model_name = LLAMA-2-7b_DIRECTORY
adapters_name = "vaishali/HiTQA-Llama"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map={"": 0}
    )
    model = PeftModel.from_pretrained(model, adapters_name)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
```

**Bengali SQL query creation**
```
python data_generation/extract_wikitables.py --table_language "bn" --data_save_path "data/bengali_tables.jsonl" --max_table_cells 500
python data_generation/create_sql_samples.py --table_language "bn" --data_save_path "data/bengali_tables.jsonl" --max_table_cells 500
python data_generation/process_code_mixed_sql.py \
                                                 --input_file "data/bengali_sql/non_numeric_code_mixed.jsonl"  \
                                                 --output_file "data/bengali_sql/non_numeric_full_indic.jsonl" \
                                                 --table_language "bn" \
                                                 --sql_language "bengali"
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

Arguments for Bengali TableQA encoder-decoder training:
```

python tableqa/train.py --pretrained_language_model "facebook/mbart-large-50" \
                --learning_rate 1e-4 \
                --train_batch_size 2 \
                --eval_batch_size 2 \
                --gradient_accumulation_steps 64 \
                --num_train_epochs 8 \
                --use_multiprocessing False \
                --num_workers 2 \
                --decoder_max_length 1024 \
                --seed 42 \
                --decoder_max_length 1024 \
                --language "bn" \
                --output_dir "experiments/banglaTabQA_mbart" 

```

Arguments for Bengali TableQA Llama training:
```
python tableqa/train.py --pretrained_language_model "llama-2-7b-hf" \
                --learning_rate 1e-4 \
                --train_batch_size 8 \
                --eval_batch_size 8 \
                --gradient_accumulation_steps 4 \
                --num_train_epochs 5 \
                --save_total_limit 50 \
                --seed 1234 \
                --warmup_ratio 0.04 \
                --use_multiprocessing False \
                --num_workers 2 \
                --decoder_max_length 1024 \
                --local_rank -1 \
                --language "bn" \
                --dataset "banglaTabQA" \
                --load_in8_bit \
                --r 8 \
                --lora_alpha 16  \
                --output_dir "experiments/bnTQA_llama_8bit_8r_alpha16" 
```

Arguments for Hindi TableQA encoder-decoder model training:
```
python tableqa/train.py --pretrained_language_model "facebook/mbart-large-50" --learning_rate 1e-4 \
                --train_batch_size 2 --eval_batch_size 2 --gradient_accumulation_steps 64 --num_train_epochs 8 \
                --use_multiprocessing False --num_workers 2 --decoder_max_length 1024  \
                 --seed 42 --decoder_max_length 1024 --language "hi" \
                --output_dir "experiments/hindiTabQA_mbart" 
```
Arguments for Hindi TableQA Llama model training:
```
python tableqa/train.py \
                --pretrained_language_model "llama-2-7b-hf" \
                --learning_rate 1e-4 \
                --train_batch_size 2 \
                --eval_batch_size 2 \
                --gradient_accumulation_steps 16 \
                --num_train_epochs 5 \
                --save_total_limit 50 \
                --seed 1234 --warmup_ratio 0.04 \
                --use_multiprocessing False \
                --num_workers 2 \
                --decoder_max_length 1024 \
                --language "hi" \
                --dataset "hindiTabQA" \
                --load_in8_bit \
                --lora_r 8 \
                --lora_alpha 16  \
                --local_rank -1 \
                --output_dir "experiments/hiTQA_llama_8bit_8r_alpha16" 
```
Arguments for Bengali Table QA evaluation:
```
python tableqa/evaluate_tableqa.py --pretrained_model_name "vaishali/BnTQA-mBart" \
                --batch_size 2 --generation_max_length 1024 \
                --validation_dataset_path "data/mbart-50_tokenized/mbart-50_validation.hf" \
                --predictions_save_path "experiments/predictions/mbart-50_validation.jsonl" 
```

**Results**


| | Tab | Row | Col | Cell | Tab | Row | Col | Cell | Tab | Row | Col | Cell | Tab | Row | Col | Cell
--------| ------- | ------------| -----------          | -------            | ------------        | -------------           |-----------             | ----------              | --------------        | ---------------    |------------ | ---------------    |------------ | ---------------    |------------  |------------
| <td colspan=8 align="center">Bengali  <td colspan=8 align="center">Hindi
Model  <td colspan=4 align="center">Validation Set scores (%) <td colspan=4 align="center">Test Set scores (%) <td colspan=4 align="center">Validation Set scores (%) <td colspan=4 align="center">Test Set scores (%)
En2(Bn/Hi) | 0.05 |3.06 |0.20 |3.07 |0.00 |4.73 |0.00 |4.73 |0.00 |3.37 |0.47 |3.43 |0.00 |5.03 |8.26 |5.03
OdiaG | 0.00 |3.89 |0.00 |3.89 |0.69 |1.77 |0.69 |1.42 |− |− |− |− |− |− |− |−
OpHathi | −| − |− |− |− |− |− |− |0.00 |0.00 |0.00 |0.00 |0.00 |0.11 |0.37 |0.74
GPT-3.5 | 1.14 |4.81 |1.67 |5.14 |6.04 |10.06 |9.12 |9.84 |4.81 |8.94 |4.99 |9.71 |8.20 |10.29 |7.10 |9.81
GPT-4 | 0.00 |13.57 |5.43 |14.65 |26.83 |38.67 |26.74 |36.51 |15.53 |22.60 |16.02 |22.25 |11.11 |21.49 |11.76 |20.84
|  <td colspan=8 align="center">BnTQA  <td colspan=8 align="center">HiTQA
-llama |60.08 |68.30 |60.47 |68.30 |9.41 |12.35 |9.85 |11.87 |14.76 |9.92 |14.13 |7.29 |13.11 |9.71 |11.11 |7.66
-mBart |56.63 |64.10 |56.79 |64.31 |35.88 |33.16 |35.88 |33.16 |92.09 |87.97 |92.02 |87.97 |33.06 |43.35 |33.88 |43.35
-M2M |45.31 | 58.07 |45.29 |58.04 |28.05 |34.55 |28.05 |34.55 |89.55 |85.32 |89.34 |85.15 |28.93 |33.11 |28.92 |33.10
-BnTQA |− |− |− |− |−|− |− |− |92.40 |88.10 |92.42 |88.12 |41.32 |47.26 |41.32 |47.26



Please cite our work if you use our code or datasets:
```
@inproceedings{pal-etal-2024-table,
    title = "Table Question Answering for Low-resourced {I}ndic Languages",
    author = "Pal, Vaishali  and
      Kanoulas, Evangelos  and
      Yates, Andrew  and
      Rijke, Maarten",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.5",
    pages = "75--92",
    abstract = "TableQA is the task of answering questions over tables of structured information, returning individual cells or tables as output. TableQA research has focused primarily on high-resource languages, leaving medium- and low-resource languages with little progress due to scarcity of annotated data and neural models. We address this gap by introducing a fully automatic large-scale tableQA data generation process for low-resource languages with limited budget. We incorporate our data generation method on two Indic languages, Bengali and Hindi, which have no tableQA datasets or models. TableQA models trained on our large-scale datasets outperform state-of-the-art LLMs. We further study the trained models on different aspects, including mathematical reasoning capabilities and zero-shot cross-lingual transfer. Our work is the first on low-resource tableQA focusing on scalable data generation and evaluation procedures. Our proposed data generation method can be applied to any low-resource language with a web presence. We release datasets, models, and code (https://github.com/kolk/Low-Resource-TableQA-Indic-languages).",
}
```
