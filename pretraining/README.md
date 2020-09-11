# Fremgangsmetode for å videretrene/trene opp BERT fra scratch:

- Link til treningsdata: [BERT-training-data](https://drive.google.com/drive/folders/10zLTJCATlc1mLZyGhO-1EBXUjGKJVdX-?usp=sharing)

### Fremgangsmåte 1:

1. Kjør egen VM eller koble til TPU på ønskelig sted, f.eks. i google's cloud console eller kaggle

2. Påse at du bruker en eldre versjon av tensorflow (koden under er fra 2018), så versjoner mellom 1.5.0 og 1.15.0 må benyttes

```
pip install tensorflow==1.5.0
```

3. Clone google BERT repo og cd inn i den

```
git clone https://github.com/google-research/bert.git

cd bert
```

4. Lag treningsdata ved å kjøre (fjern gs om du ikke har det i Google Storage Bucket).

- NB: la max_predictions_per_seq være omtrent lik max_seq_length * masked_lm_prob (256*0.15 er ish 40). 
- NB: ved videretrening bør do_lower_case matche det modellen er trent opp på tidligere
- NB: max_seq_length må ikke være like stor som det modellen er trent opp på tidligere, men bør være en faktor på 64.

```
! python create_pretraining_data.py \
  --input_file=gs://din_input_fil.txt \
  --output_file=gs://output_fil.tfrecord \
  --vocab_file=gs://PATH_TO_VOCAB_FIL/vocab.txt \
  --do_lower_case=False \
  --max_seq_length=256 \
  --max_predictions_per_seq=40 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

5. Fortsett trening fra forrige lagret checkpoints.

- NB: max_seq_length, max_predictions_per_seq må matche det .tfrecord filene ble trent opp på.
- NB: use_tpu = True benyttes naturligvis dersom en benytter seg av en tpu
- NB: learning_rate bør være 2e-5 for videretrening og høyere ved trening fra skratch (google anbefaler x*e-4)
- NB: train_batch_size kan godt økes, spesielt om det trenes opp på TPU (sjekk docs til tpu provider for tips).

```
  python run_pretraining.py \
  --input_file=gs://din_input_fil.tfrecord \
  --output_dir=gs://output_folder/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=gs://PATH_TO_MODELL/multi_cased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=gs://PATH_TO_MODELL/multi_cased_L-12_H-768_A-12/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=256 \
  --max_predictions_per_seq=40 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 \
  --use_tpu=True \
  --tpu_name=test-tpu
```

6. Husk å cleane opp de ressursene en bruker etterpå (spesielt om en kjører på TPU!)

Feks for google sine tpu-er
```
  exit
  
  ctpu delete --name=test-tpu

  ctpu status ### Skal ikke være kjørende nå
```

7. Dette lagrer modellen som .ckpt filer, anbefales deretter å gjøre om til PyTorch modell eller TF modell. Her er script for å gjøre om til PyTorch:

```
import torch
from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert

"""
  Params:
    - ckpt_path, f.eks. bert_model.ckpt (Merk at filendelsen må være .ckpt, og ikke noe mer! Peker til alle 3 filene som lagres over).
    - config_file, peker til bert_config.json fil (Må ha .json endelse)
    - pytorch_dump_path, f.eks. pytorch_model.bin (Må ha .bin endelse, legg til tom folder)
    
    - NB, config_file og .ckpt filene må ligge i samme folder
    
"""
def convert_ckpt_to_pytorch(ckpt_path, config_file, pytorch_dump_path):
  config = BertConfig.from_json_file(config_file)
  model = BertForPreTraining(config)
  
  load_tf_weights_in_bert(model, config, ckpt_path)
  
  torch.save(model.state_dict(), pytorch_dump_path)
```
