# LowResourceTableQA
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
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, src_lang="bn", tgt_lang="bn")
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

Please cite our work if you use our code or datasets:
```
@misc{pal2024tablequestionansweringlowresourced,
      title={Table Question Answering for Low-resourced Indic Languages}, 
      author={Vaishali Pal and Evangelos Kanoulas and Andrew Yates and Maarten de Rijke},
      year={2024},
      eprint={2410.03576},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.03576}, 
}
```
