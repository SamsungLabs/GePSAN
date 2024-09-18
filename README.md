# Code for GePSAn: Generative Procedure Step Anticipation in Cooking Videos
 
## Data Preparation
### Getting UniVL Features
Get and normalize the output embeddings of the Text Encoder and Video Encoder of the UniVL model:
Paper: https://arxiv.org/pdf/2002.06353
Code: https://github.com/microsoft/UniVL

## Training
### Pretraining on Recipe1M
Change the dataset paths and the hyperparameters from configs/config.yaml
python train_recipe1M_gepsan.py --num_workers 12 --batch_size 50 --learning_rate 0.0001 --num_epochs 50 --wandb_log true --seed 1 --run_id "an experiment identifier"  

### Finetuning on YouCookII
Change the dataset paths and the hyperparameters from configs/config_ycii.yaml
python train_youcookii_gepsan.py --num_workers 12 --batch_size 50 --learning_rate 0.0001 --num_epochs 10 --wandb_log true --seed 1 --features_type visual --run_id "an experiment identifier" --pretrained_model_folder "directory where the pretrained model is" --pretrained_model_suffix 'best' --split_type unseen_split

split_type is "unseen_split" or "seen_split" as in Table 1 and 2 in the paper, or "original_split" as in Table 7 in the appendix
set num_epochs to 0 to get the zero shot performance of the model
features_type is "visual" for the UniVL visual features of the videos, or "textual" for the UniVL textual features of the instructions


## Citation
@InProceedings{Abdelsalam_2023_ICCV,
    author    = {Abdelsalam, Mohamed A. and Rangrej, Samrudhdhi B. and Hadji, Isma and Dvornik, Nikita and Derpanis, Konstantinos G. and Fazly, Afsaneh},
    title     = {GePSAn: Generative Procedure Step Anticipation in Cooking Videos},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {2988-2997}
}