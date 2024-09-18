# GePSAn: Generative Procedure Step Anticipation in Cooking Videos

This repository contains the code for **GePSAn**, a model designed for anticipating procedural steps in cooking videos using generative techniques.

## Data Preparation

### Extracting UniVL Features
Extract and normalize the output embeddings of the Text Encoder and Video Encoder from the UniVL model, follow the code and model provided in the UniVL repository:

- **Paper**: [UniVL: A Unified Video and Language Pre-training Model](https://arxiv.org/pdf/2002.06353)
- **Code**: [Microsoft/UniVL GitHub Repository](https://github.com/microsoft/UniVL)

## Training

### Pretraining on Recipe1M
To pretrain the model on the Recipe1M dataset:

1. Modify dataset paths and hyperparameters in `configs/config.yaml`.
2. Run the pretraining script:

    ```bash
    python train_recipe1M_gepsan.py --num_workers 12 --batch_size 50 --learning_rate 0.0001 --num_epochs 50 --wandb_log true --seed 1 --run_id "experiment_identifier"
    ```

### Finetuning on YouCookII
To finetune the model on the YouCookII dataset:

1. Update dataset paths and hyperparameters in `configs/config_ycii.yaml`.
2. Run the finetuning script:

    ```bash
    python train_youcookii_gepsan.py --num_workers 12 --batch_size 50 --learning_rate 0.0001 --num_epochs 10 --wandb_log true --seed 1 --features_type visual --run_id "experiment_identifier" --pretrained_model_folder "path_to_pretrained_model" --pretrained_model_suffix 'best' --split_type unseen_split
    ```

- **`split_type`** options:
  - `unseen_split` or `seen_split` as described in Tables 1 and 2 of the paper.
  - `original_split` as described in Table 7 of the appendix.

- Set `num_epochs` to `0` for zero-shot evaluation of the model.

- **`features_type`** options:
  - `visual` for using UniVL visual features from videos.
  - `textual` for using UniVL textual features from cooking instructions.

## Citation

If you find this work useful, please consider citing the following paper:
```
@InProceedings{Abdelsalam_2023_ICCV,
    author    = {Abdelsalam, Mohamed A. and Rangrej, Samrudhdhi B. and Hadji, Isma and Dvornik, Nikita and Derpanis, Konstantinos G. and Fazly, Afsaneh},
    title     = {GePSAn: Generative Procedure Step Anticipation in Cooking Videos},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {2988-2997}
}
```
