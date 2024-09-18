import logging
import os

import torch
import torch.nn as nn
import wandb
from Vocabulary import Vocabulary
import nltk


## Logging ##
# The arguments that wandb should consider as hyperparameters
HYPERPARAMETERS = ["features_type", "word_dim", "sentEnd_hiddens", "recipe_inDim", "recipe_nlayers", "recipe_hiddens"
                   "recipe_nheads", "sentDec_hiddens", "sentDec_nlayers", "sentDec_nheads", "num_epochs", "batch_size",
                   "learning_rate", "teach_force_ep", "alpha", "beta", "gamma", "temperature",
                   "freeze_recipe", "freeze_encoder", "freeze_decoder", "freeze_verbs", "pretrained_model_suffix",
                   "run_id", "seed", "beta_1", "beta_2", "weight_decay", "sigma",
                   "cvae_latent_dim", "cvae_hidden_dim", "cvae_n_hidden_layers", "cvae_concat_next", "final_kl_weight",
                   "kl_annealing_steps", "learnable_prior", "split_type", "nucleus_sampling"]


def get_model_params(args, identifier=""):
    param_all = ('e' + str(args.recipe_inDim) + '_lre' + str(args.recipe_nlayers) +
                 '_hd' + str(args.sentDec_hiddens) + '_b' + str(args.batch_size) +
                 '_l' + str(args.learning_rate).replace(".", "_") + '_s' + str(args.seed))
    if identifier != "":
        param_all += f"_{identifier}"
    param_all += f'_{args.run_id}'
    return param_all


def initialize_wandb(args, wandb, model_folder, job_type, wandb_id=None, project="zero_shot_cvae"):
    if wandb_id is not None:
        wandb.init(project=project, group=model_folder, job_type=job_type,
                   config={key: value for key, value in vars(args).items() if key in HYPERPARAMETERS},
                   id=args.wandb_id, resume="allow", allow_val_change=True)
    else:
        wandb.init(project=project, group=model_folder, job_type=job_type,
                   config={key: value for key, value in vars(args).items() if key in HYPERPARAMETERS})

    wandb.define_metric("epoch")
    wandb.define_metric("interval/*", step_metric="interval")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric(f"metrics/*", step_metric="epoch")
    return wandb


def wandb_log_epoch_metrics(epoch, train_logs: dict = None, val_logs: dict = None, **kwargs):
    logs = {'epoch': epoch}
    logs.update(kwargs)
    if train_logs is not None:
        logs.update({'train/' + key: value for key, value in train_logs.items()})
    if val_logs is not None:
        logs.update({'val/' + key: value for key, value in val_logs.items()})
    wandb.log(logs)


def save_model(args, model, epoch_val, model_suffix="", save_to_wandb=False, force_sync=False):
    if epoch_val == 0:
        num_epochs = ''
    else:
        num_epochs = '-' + str(epoch_val)
    if model_suffix != "":
        model_suffix = '-' + model_suffix

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    saving_paths = [args.model_path]
    if save_to_wandb:
        saving_paths.append(wandb.run.dir)

    for saving_path in saving_paths:
        torch.save(model.state_dict(),
                   os.path.join(saving_path, f'model{model_suffix}{num_epochs}.ckpt'))
    if save_to_wandb and force_sync:
        wandb.save(wandb.run.dir)


def load_model(args, model, model_suffix="", rank=None):
    if rank is not None:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    else:
        map_location = None

    # Load the trained model parameters
    if model_suffix != "":
        model_suffix = '-' + str(model_suffix)
    model.load_state_dict(
        torch.load(os.path.join(args.model_path, f'model{model_suffix}.ckpt'), map_location=map_location))
    logging.info(f"Pretrained Models Loaded on device {rank if rank is not None else 0}")


## Evaluation ##
def generate(vocab, sent_ids, prefix="pred"):
    pred_sentence = ids2words(vocab, sent_ids[0].cpu().numpy())
    msg = f'{prefix}: {pred_sentence.strip()}'
    logging.info(msg)
    return msg


def ids2words(vocab, target_ids):
    target_caption = []
    for word_id in target_ids:
        word = vocab.idx2word[word_id]
        if word == '<start>':
            continue
        if word == '<end>':
            break
        target_caption.append(word)
    target_sentence = ' '.join(target_caption)
    return target_sentence


def truncate(sen, end_token=30169):
    try:
        end = (sen == end_token).nonzero(as_tuple=True)[0][0]
        result = sen[:end]
    except:
        result = sen
    return result


## Training ##
def generate_padding_mask(rec_lens, device, max_len=None) -> torch.BoolTensor:
    if max_len is None:
        max_len = max(rec_lens)
    if not isinstance(rec_lens, torch.Tensor):
        rec_lens = torch.tensor(rec_lens)
    padding_mask = torch.arange(max_len).expand(rec_lens.shape[0], max_len).ge(rec_lens.unsqueeze(1))
    return padding_mask.to(device)


def generate_causal_mask(max_len, device, starting_idx=1) -> torch.FloatTensor:
    """
    args:
        starting_idx: The starting index of the causal mask
            example: if the starting index is set to 1, then the first input would be visible in the first step, the
            first two inputs would be visible in the second step, etc.
            However, if the starting index is set to 5, then all the first five inputs would be visible during the first
            five steps, the first six inputs would be visible in the sixth step, etc.
    """
    assert starting_idx > 0, "The starting index starts from 1"
    causal_mask = torch.triu(torch.full((max_len, max_len), float('-inf')), diagonal=1)
    if starting_idx > 1:
        causal_mask[:, :starting_idx] = 0
    return causal_mask.to(device)


def generate_output_mask(input_mask: torch.FloatTensor, padding_mask: torch.BoolTensor) -> torch.BoolTensor:
    """
    a shifted version of the input mask (since masking an instruction means generating that instruction in the previous
    step)
    args:
        input_mask: An input mask with the masked tokens as -inf and the unmasked tokens as 0 [recipe length,
            recipe length] or [batch size, recipe length] or [batch size, recipe length, recipe length]
        padding_mask: A padding mask with the padding as True. [batch, recipe length]
    return:
        output_mask: A 1-D boolean mask with the masked steps as True and the unmasked steps as False [batch x
            recipe lengths]. The masked steps are the ones that need to be generated.
    """
    b_size = padding_mask.shape[0]
    seq_len = padding_mask.shape[1]
    output_mask = torch.ones_like(padding_mask).bool()
    if input_mask.shape == (seq_len, seq_len):
        output_mask[:, :-1] = input_mask.diagonal(1).bool()
        output_mask = output_mask[~padding_mask]
    elif input_mask.shape == (b_size, seq_len):
        output_mask[:, :-1] = input_mask[:, 1:].bool()
        output_mask = output_mask[~padding_mask]
    elif input_mask.shape == (b_size, seq_len, seq_len):
        output_mask[:, :-1] = input_mask.diagonal(1, dim1=1, dim2=2).bool()
        output_mask = output_mask[~padding_mask]
    else:
        raise ValueError(f"Unrecognized shape of input_mask {input_mask.shape}")

    if not output_mask.any():
        logging.warning("Empty output mask, setting the last step in the batch to True")
        output_mask[-1] = True
    return output_mask
