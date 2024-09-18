# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import pickle
import random
import sys
import time

import numpy as np
import regex as re
import torch.nn.functional as F
import yaml
from torch.nn.utils.rnn import pack_padded_sequence

from dataset.dataloader_ycii import get_youcookii_loader
from model.model import GEPSAN
from utils.evaluate import Evaluator, EvaluatorMacro
from utils.utils import *
from Vocabulary import Vocabulary

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_TRAIN_INSTRUCTION_LENGTH = 100


def train(args, test_split=0):
    # Build the models
    model = GEPSAN(args).to(device)
    if args.pretrained_model_suffix is not None:
        load_model(args, model, args.pretrained_model_suffix)
    if args.copy_textual_encoder_to_visual:
        model.copy_textual_encoder_to_visual()

    # Loss and optimizer
    criterion_sent = nn.CrossEntropyLoss()
    if args.freeze_encoder:
        model.freeze_modality_encoder()
    if args.freeze_recipe:
        model.freeze_recipe_encoder()
    if args.freeze_decoder:
        model.freeze_instruction_decoder()
    params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay,
                                  betas=(args.beta_1, args.beta_2))

    # Build data loader
    with open(args.vocab_bin, 'rb') as f:
        vocab = pickle.load(f, encoding='latin1')
    val_loader = get_youcookii_loader(args, args.batch_size, vocab, shuffle=False, num_workers=args.num_workers,
                                      train=False, test_split=test_split, split_type=args.split_type)
    train_loader = get_youcookii_loader(args, args.batch_size, vocab, shuffle=True, num_workers=args.num_workers,
                                        train=True, test_split=test_split, split_type=args.split_type)

    # Prepare the scheduler
    total_step = len(train_loader)

    def scheduler_lambda(step):
        # Linear ascend followed by exponential decay
        ascend_cycle_ep = 1
        ascend_cycle = ascend_cycle_ep * total_step
        ascend_stage = step <= ascend_cycle
        if ascend_stage:
            return step / ascend_cycle
        else:
            return 1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_lambda)

    best_meteork_metric = 0
    best_scores = None

    if args.features_type == "textual":
        visual_modality = False
    elif args.features_type == "visual":
        visual_modality = True

    for epoch in range(-1, args.num_epochs):
        start_tr = time.time()
        epoch_loss = 0

        # Train
        if epoch > -1:  # the first epoch is only for evaluationg the recipe1m pretrained model on YCII
            model.train()
            for i, (ingredients_v, rec_lens, sentences_v, sent_lens, univl_feats_v) in enumerate(train_loader):
                ingredients_v = ingredients_v.to(device)  # [N, Nv] -> Nv = ingredient vocab. len
                sentences_v = sentences_v.to(device)  # [Nb, Ns] -> [total num sent, max sent len.]
                sentences_v = sentences_v[:, :MAX_TRAIN_INSTRUCTION_LENGTH]
                sent_lens = torch.clamp(sent_lens, max=MAX_TRAIN_INSTRUCTION_LENGTH)
                if args.features_type == "textual":
                    univl_feats_v = univl_feats_v[0].to(device)  # [Nb, Ns] -> [total num sent, univl embeddings size]
                    assert visual_modality == False
                elif args.features_type == "visual":
                    univl_feats_v = univl_feats_v[1].to(device)  # [Nb, Ns] -> [total num sent, univl embeddings size]
                    assert visual_modality == True
                else:
                    raise f"features_type {args.features_type} is unknown"

                generated_instr_embeds, input_instr_embeds, kl_loss = model(ingredients_v, univl_feats_v, rec_lens,
                                                                            visual_modality=visual_modality)

                sentence_target = pack_padded_sequence(sentences_v, sent_lens, batch_first=True, enforce_sorted=False)[
                    0]  # [ sum(sent_lens) ]

                """ Compute the losses """
                aux_loss = F.mse_loss(generated_instr_embeds, input_instr_embeds.detach(), reduction='mean')

                # reconstructed instructions
                sentence_rec = model.decode_embeddings(input_instr_embeds, sentences_v, sent_lens)
                rec_loss = criterion_sent(sentence_rec, sentence_target)

                # Predicted instructions
                sentence_pred = model.decode_embeddings(generated_instr_embeds, sentences_v, sent_lens)
                pred_loss = criterion_sent(sentence_pred, sentence_target)

                beta = min(args.final_kl_weight,
                           (i + epoch * total_step) / args.kl_annealing_steps * args.final_kl_weight)
                beta = max(0.00001, beta)
                loss = pred_loss + args.gamma * rec_loss + args.alpha * aux_loss + beta * kl_loss

                epoch_loss += loss.item() / len(train_loader)

                """ Backpropagation """
                model.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                """ Printing and evaluations """
                if i % args.log_step == 0:  # Print log info
                    logging.info(
                        (f'Epoch {epoch}/{args.num_epochs}, Step {i}/{total_step}, '
                         f'Reconstruction Loss: {rec_loss.item():.4f}, '
                         f'Prediction Loss: {pred_loss.item():.4f}, Auxiliary loss: {aux_loss.item():.4f}, '
                         f'KL loss: {kl_loss.item():.4f}, Total Loss: {loss.item():.4f}, '
                         f'Learning_rate: {scheduler.get_last_lr()[0]}, Beta: {beta}')
                    )
                    if args.wandb_log:
                        wandb.log(
                            {f'interval/Total Loss': loss.item(), 'interval/Prediction Loss': pred_loss.item(),
                             'interval/Reconstruction Loss': rec_loss.item(), 'interval/KL Loss': kl_loss.item(),
                             'interval/Auxiliary Loss': aux_loss.item(), 'interval/Beta': beta,
                             'interval/Learning_rate': scheduler.get_last_lr()[0], 'interval': i + epoch * total_step}
                        )

                    # Generate Samples
                    if i % (args.log_step * 10) == 0:  # Print qualitative samples
                        logging.info("Training Recipe Samples:")
                        samples_to_print = random.choices(range(len(rec_lens)), k=3)
                        for j, rec_id in enumerate(samples_to_print):
                            logging.info(f"\nSample Recipe {j + 1}:")
                            instr2gen_id = random.randrange(1, rec_lens[rec_id])
                            n_instr_to_print = min(rec_lens[rec_id], instr2gen_id + 3)
                            for instr_id in range(n_instr_to_print):
                                instr_id_abs = instr_id + sum(rec_lens[:rec_id])
                                gt_sentence = ids2words(vocab, sentences_v[instr_id_abs, :].cpu().numpy())
                                if instr_id != instr2gen_id:
                                    logging.info(f'Instruction {instr_id}: {gt_sentence}')
                                elif instr_id == instr2gen_id:
                                    logging.info(f'===> Instruction {instr_id}: {gt_sentence}')
                                    with torch.no_grad():
                                        generated_sent_ids = model.decode_embeddings_greedy(
                                            generated_instr_embeds[instr_id_abs].unsqueeze(0))
                                        generate(vocab, generated_sent_ids, f"===> Generated Instruction {instr_id}")

        end_tr = time.time()

        # Evaluate on the validation set
        epoch_loss_val = 0
        start_val = time.time()
        model.eval()
        k_samples = 5
        if args.macro:
            metrics_evaluator = EvaluatorMacro(vocab, args.vocab_ing, args.vocab_verb, best_k=k_samples)
        else:
            metrics_evaluator = Evaluator(vocab, args.vocab_ing, args.vocab_verb, best_k=k_samples)
        # result, gt_v = [], []

        with torch.no_grad():
            for i, (ingredients_v, rec_lens, sentences_v, sent_lens, univl_feats_v) in enumerate(val_loader):
                ingredients_v = ingredients_v.to(device)  # [N, Nv] -> Nv = ingredient vocab. len
                sentences_v = sentences_v.to(device)  # [Nb, Ns] -> [total num sent, max sent len.]
                if args.features_type == "textual":
                    univl_feats_v = univl_feats_v[0].to(device)  # [Nb, Ns] -> [total num sent, univl embeddings size]
                elif args.features_type == "visual":
                    univl_feats_v = univl_feats_v[1].to(device)  # [Nb, Ns] -> [total num sent, univl embeddings size]
                else:
                    raise f"features_type {args.features_type} is unknown"

                generated_instr_mean, generated_instr_sampled, (_, generated_instr_embeds_sampled) = model.generate(
                    ingredients_v, univl_feats_v, rec_lens, visual_modality=visual_modality, n_samples=k_samples)

                sentence_target = pack_padded_sequence(sentences_v, sent_lens, batch_first=True, enforce_sorted=False)[
                    0]  # [ sum(sent_lens) ]

                """ Compute the loss """
                # Generate a predicted instruction using teacher forcing and one sampled instruction embedding to
                # compute the validation loss
                sentence_dec = model.decode_embeddings(generated_instr_embeds_sampled[:, 0], sentences_v, sent_lens)
                epoch_loss_val += criterion_sent(sentence_dec, sentence_target).item() / len(val_loader)

                # Jaccard similarity
                senternces_v_set = [set(x.item() for x in truncate(sentences_v[i])) for i in
                                    range(sentences_v.shape[0])]
                generated_instr_sampled_set = [
                    [set(x.item() for x in truncate(generated_instr_sampled[i, j])) for j in
                     range(generated_instr_sampled.shape[1])] for i in range(generated_instr_sampled.shape[0])]
                intersection = [torch.tensor([len(
                    generated_instr_sampled_set[j][k].intersection(senternces_v_set[j])) / len(
                    generated_instr_sampled_set[j][k].union(senternces_v_set[j])) for k in range(k_samples)]) for j
                                in range(sentences_v.shape[0])]
                intersection = torch.cat([item.unsqueeze(0) for item in intersection], dim=0)
                best_indices = intersection.argmax(dim=1)
                best_generated_instr_sampled = generated_instr_sampled[torch.arange(len(best_indices)), best_indices]
                metrics_evaluator.extend_gts_res(generated_instr_mean, best_generated_instr_sampled, sentences_v)

                # Generate Samples
                logging.info("Validation Recipe Samples:")
                samples_to_print = random.choices(range(len(rec_lens)), k=3)
                for j, rec_id in enumerate(samples_to_print):
                    logging.info(f"\nSample Recipe {j + 1}:")
                    instr2gen_id = random.randrange(1, rec_lens[rec_id])
                    n_instr_to_print = min(rec_lens[rec_id], instr2gen_id + 3)
                    for instr_id in range(n_instr_to_print):
                        instr_id_abs = instr_id + sum(rec_lens[:rec_id])
                        gt_sentence = ids2words(vocab, sentences_v[instr_id_abs, :].cpu().numpy())
                        if instr_id != instr2gen_id:
                            logging.info(f'Instruction {instr_id}: {gt_sentence}')
                        elif instr_id == instr2gen_id:
                            logging.info(f'===> Instruction {instr_id}: {gt_sentence}')
                            n_samples2print = min(k_samples, 3)
                            with torch.no_grad():
                                for k in range(n_samples2print):
                                    generated_sent_ids = generated_instr_sampled[instr_id_abs, k].unsqueeze(0)
                                    generate(vocab, generated_sent_ids, f"===> Generated Instruction {instr_id}")

        metrics_evaluator.calculate_scores()
        end_val = time.time()
        """ Printing and evaluations """
        if f'meteor_best{k_samples}' in metrics_evaluator.current_scores:
            metric = f'meteor_best{k_samples}'
        else:
            metric = f'Mmeteor_best{k_samples}'
        reference_meteor = metrics_evaluator.current_scores[metric]

        if reference_meteor >= best_meteork_metric:
            best_epoch = epoch
            best_meteork_metric = reference_meteor
            save_model(args, model, "best", args.model_file_suffix,
                       save_to_wandb=args.wandb_save_model if args.wandb_log else None)
            if args.wandb_log:
                wandb.run.summary[f"best_epoch/train_prediction_loss"] = epoch_loss
                wandb.run.summary["best_epoch"] = best_epoch
                wandb.run.summary[f"best_epoch/val_prediction_loss"] = epoch_loss_val
                wandb.run.summary[f"best_epoch/val_{metric}"] = reference_meteor
            best_scores = metrics_evaluator.current_scores

        logging.info(
            (f'Train Epoch {epoch}/{args.num_epochs}, Total Loss: {epoch_loss:.4f}, '
             f'Duration: {end_tr - start_tr:.1f}')
        )
        logging.info(
            (f'Validation Epoch {epoch}/{args.num_epochs}, Total Loss: {epoch_loss_val:.4f}, '
             f'Duration: {end_val - start_val:.1f}')
        )
        message = [f"Metrics"]
        for key, val in metrics_evaluator.current_scores.items():
            message.append(f"{key}: {val:.4f}")
        logging.info(' '.join(message))

        if args.wandb_log:
            metrics_ = metrics_evaluator.current_scores
            metrics = {f"metrics/{key}": val for key, val in metrics_.items()}
            metrics.update({'val_epoch_duration': end_val - start_val})
            metrics.update({'train_epoch_duration': end_tr - start_tr})
            wandb_log_epoch_metrics(
                epoch=epoch,
                train_logs={f"prediction_loss": epoch_loss},
                val_logs={f"prediction_loss": epoch_loss_val, "best_metric": best_meteork_metric},
                **metrics
            )
    # Save the final model
    save_model(args, model, args.num_epochs, args.model_file_suffix,
               save_to_wandb=args.wandb_save_model if args.wandb_log else None)
    if args.wandb_log:
        wandb.finish()
    return best_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    name_repo = 'gepsan'

    parser.add_argument('--results_root', type=str, default='results/',
                        help='root path for logging and checkpoints')
    parser.add_argument('--config', type=str, default='configs/config_ycii.yaml')

    parser.add_argument('--run_id', type=str, default=None, help='')
    parser.add_argument('--pretrained_model_folder', type=str, default=None,
                        help='if not provided, the experiment folder will be chosen based on the parameters provided')
    parser.add_argument('--pretrained_model_suffix', type=str, default=None, help='')
    parser.add_argument('--features_type', choices=["textual", "visual"], default="textual", help='')

    # training parameters
    parser.add_argument('--log_step', type=int, default=20, help='step size for printing log info')
    parser.add_argument('--save_step', type=int, default=45, help='step size for saving trained models')
    parser.add_argument('--wandb_log', type=str, choices=["true", "false"], default="false",
                        help='log metrics to wandb')
    parser.add_argument('--wandb_save_model', action="store_true", help='save the best and last models to wandb')
    parser.add_argument('--wandb_id', type=str, default=None, help='a unique id for the run, to be used with inference')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=0, help="the random seed")
    parser.add_argument('--copy_textual_encoder_to_visual', action='store_true')
    parser.add_argument('--freeze_recipe', action="store_true",
                        help="freeze the recipe encoder")
    parser.add_argument('--freeze_encoder', action="store_true",
                        help="freeze the input encoder")
    parser.add_argument('--freeze_decoder', action="store_true",
                        help="freeze the output decoder")
    parser.add_argument('--split_type', type=str, choices=["unseen_split", "seen_split", "original_split"],
                        default="unseen_split",
                        help="use the original train/validation splits, the seen splits, or the unseen splits. Check "
                             "the paper for more info")
    parser.add_argument('--macro', action='store_true',
                        help="get the average of bleu and meteor scores after computing them per instructions")

    args = parser.parse_args()

    if args.wandb_log == "true":
        args.wandb_log = True
        import wandb
    elif args.wandb_log == "false":
        args.wandb_log = False
    else:
        raise ValueError("Unknown wandb_log value")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    args.__dict__.update(config)

    if args.pretrained_model_folder is not None and args.pretrained_model_suffix is None:
        raise ValueError("please provide the pretrained_model_suffix")

    if args.wandb_save_model and not args.wandb_log:
        raise ValueError("Cannot set wandb_save_model without setting wandb_log")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.run_id is None:
        args.run_id = str(random.randint(0, 10000))

    param_all = get_model_params(args, identifier="gepsan")
    args.param_all = param_all
    if args.pretrained_model_folder is None:
        model_folder = 'models_' + param_all
    else:
        model_folder = args.pretrained_model_folder

    results_folder = os.path.join(args.results_root, name_repo)
    args.model_path = os.path.join(args.results_root, model_folder)

    if args.pretrained_model_suffix:
        args.model_file_suffix = args.pretrained_model_suffix + '_YC2_' + args.features_type
    else:
        args.model_file_suffix = 'YC2_' + args.features_type

    freeze = args.freeze_recipe or args.freeze_encoder or args.freeze_decoder
    assert not (args.freeze_recipe and args.freeze_encoder and args.freeze_decoder), "Nothing Trainable!"
    if freeze:
        args.model_file_suffix += "_frozen"
        if args.freeze_encoder:
            args.model_file_suffix += "_encoder"
        if args.freeze_recipe:
            args.model_file_suffix += "_recipe"
        if args.freeze_decoder:
            args.model_file_suffix += "_decoder"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.model_path, "logs_" + args.model_file_suffix + ".log"), mode='a+'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Configure wandb
    if args.wandb_log:
        initialize_wandb(args, wandb, model_folder, f"Youcook2_{args.features_type}", wandb_id=args.wandb_id)

    logging.info(f"Run name: {model_folder}")
    logging.info(f"Args: {vars(args)}")
    if args.split_type == "original_split":
        train(args)
    else:
        best_scores = []
        for i in range(4):
            args.split = i
            best_scores.append(train(args, test_split=i))

        message = [f"Metrics"]
        for key in best_scores[0].keys():
            message.append(f"{key}: {sum([scores[key] for scores in best_scores]) / len(best_scores):.4f}")
        logging.info(' '.join(message))
