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

from dataset.dataloader_text import get_recipe1m_loader
from model.model import GEPSAN
from utils.evaluate import Evaluator
from utils.utils import *
from Vocabulary import Vocabulary

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_TRAIN_INSTRUCTION_LENGTH = 100


def train(args, eval_only=False):
    # Build the models
    model = GEPSAN(args).to(device)
    if args.checkpoint_model_suffix is not None:
        load_model(args, model, args.checkpoint_model_suffix)

    # Loss and optimizer
    criterion_sent = nn.CrossEntropyLoss()
    params = model.parameters()
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay,
                                  betas=(args.beta_1, args.beta_2))

    # Build data loader
    with open(args.vocab_bin, 'rb') as f:
        vocab = pickle.load(f, encoding='latin1')
    val_loader = get_recipe1m_loader(args, 250, vocab, shuffle=False, num_workers=args.num_workers, seed=args.seed,
                                     train=False)
    if not eval_only:
        train_loader = get_recipe1m_loader(args, args.batch_size, vocab, shuffle=True, num_workers=args.num_workers,
                                           seed=args.seed, train=True)

        # Prepare the scheduler
        total_step = len(train_loader)

        def scheduler_lambda(step):
            # Linear warm up
            ascend_cycle_ep = 1
            ascend_cycle = ascend_cycle_ep * total_step
            ascend_stage = step <= ascend_cycle
            if ascend_stage:
                return step / ascend_cycle
            else:
                return 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_lambda)

    best_meteork_metric = 0

    # if only evaluating, run for one epoch
    if eval_only:
        args.num_epochs = args.start_epoch + 1
    for epoch in range(args.start_epoch, args.num_epochs):
        # Train
        if not eval_only:
            start_tr = time.time()
            epoch_loss = 0
            model.train()
            for i, (ingredients_v, rec_lens, sentences_v, sent_lens, univl_feats_v) in enumerate(train_loader):
                ingredients_v = ingredients_v.to(device)  # [N, Nv] -> Nv = ingredient vocab. len
                sentences_v = sentences_v.to(device)  # [Nb, Ns] -> [total num sent, max sent len.]
                sentences_v = sentences_v[:, :MAX_TRAIN_INSTRUCTION_LENGTH]
                sent_lens = torch.clamp(sent_lens, max=MAX_TRAIN_INSTRUCTION_LENGTH)
                univl_feats_v = univl_feats_v.to(device)  # [Nb, Ns] -> [total num sent, univl embeddings size]

                generated_instr_embeds, input_instr_embeds, kl_loss = model(ingredients_v, univl_feats_v, rec_lens,
                                                                            visual_modality=False)

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

                kl_ep = epoch - 1  # ignore the first epoch for kl annealing as it is a warm up epoch
                beta = min(args.final_kl_weight,
                           (i + kl_ep * total_step) / args.kl_annealing_steps * args.final_kl_weight)
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
        metrics_evaluator = Evaluator(vocab, args.vocab_ing, args.vocab_verb, best_k=k_samples)

        with torch.no_grad():
            for i, (ingredients_v, rec_lens, sentences_v, sent_lens, univl_feats_v) in enumerate(val_loader):
                ingredients_v = ingredients_v.to(device)  # [N, Nv] -> Nv = ingredient vocab. len
                sentences_v = sentences_v.to(device)  # [Nb, Ns] -> [total num sent, max sent len.]
                univl_feats_v = univl_feats_v.to(device)  # [Nb, Ns] -> [total num sent, univl embeddings size]

                generated_instr_mean, generated_instr_sampled, (_, generated_instr_embeds_sampled) = model.generate(
                    ingredients_v, univl_feats_v, rec_lens, visual_modality=False, n_samples=k_samples)

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
                    n_instr_to_print = min(rec_lens[rec_id], instr2gen_id+3)
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

        if (reference_meteor >= best_meteork_metric and not eval_only) or eval_only:
            best_epoch = epoch
            best_meteork_metric = reference_meteor
            if not eval_only:
                model.copy_textual_encoder_to_visual()
                save_model(args, model, "best", save_to_wandb=args.wandb_save_model, force_sync=True)
            if args.wandb_log:
                if not eval_only:
                    wandb.run.summary[f"best_epoch/train_prediction_loss"] = epoch_loss
                wandb.run.summary["best_epoch"] = best_epoch
                wandb.run.summary[f"best_epoch/val_prediction_loss"] = epoch_loss_val
                wandb.run.summary[f"best_epoch/val_{metric}"] = reference_meteor

        if not eval_only:
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

        if (epoch + 1) % 5 == 0 and not eval_only:  # Save the model checkpoints
            model.copy_textual_encoder_to_visual()
            save_model(args, model, epoch + 1)
        if args.wandb_log:
            metrics_ = metrics_evaluator.current_scores
            metrics = {f"metrics/{key}": val for key, val in metrics_.items()}
            metrics.update({'val_epoch_duration': end_val - start_val})
            if not eval_only:
                metrics.update({'train_epoch_duration': end_tr - start_tr})
            wandb_log_epoch_metrics(
                epoch=epoch,
                train_logs={f"prediction_loss": epoch_loss} if not eval_only else {},
                val_logs={f"prediction_loss": epoch_loss_val, "best_metric": best_meteork_metric},
                **metrics
            )
    # Save the final model
    if not eval_only:
        model.copy_textual_encoder_to_visual()
        save_model(args, model, args.num_epochs, save_to_wandb=args.wandb_save_model, force_sync=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    name_repo = 'gepsan'

    parser.add_argument('--results_root', type=str, default='results/',
                        help='root path for logging and checkpoints')
    parser.add_argument('--config', type=str, default='configs/config.yaml')

    parser.add_argument('--run_id', type=str, default=None, help='')
    parser.add_argument('--eval_only', action="store_true", help= "Only evaluate, don't train")
    parser.add_argument('--specific_model_path', type=str, default=None,
                        help='overwrites the model path generated from the hyperparameters. Not needed for '
                             'resuming the training.')
    # training parameters
    parser.add_argument('--log_step', type=int, default=20, help='step size for printing log info')
    parser.add_argument('--save_step', type=int, default=45, help='step size for saving trained models')
    parser.add_argument('--wandb_log', type=str, choices=["true", "false"], default="true", help='log metrics to wandb')
    parser.add_argument('--wandb_save_model', action="store_true", help='save the best and last models to wandb')
    parser.add_argument('--wandb_id', type=str, default=None, help='a unique id for the run, to be used with inference')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=0, help="the random seed")

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

    if args.wandb_save_model and not args.wandb_log:
        raise ValueError("Cannot set wandb_save_model without setting wandb_log")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.run_id is None:
        args.run_id = str(random.randint(0, 10000))

    param_all = get_model_params(args, identifier="gepsan")
    args.param_all = param_all
    model_folder = 'models_' + param_all
    results_folder = os.path.join(args.results_root, name_repo)
    args.model_path = os.path.join(args.results_root, model_folder)
    if args.specific_model_path is not None:
        args.model_path = args.specific_model_path

    args.start_epoch = 0
    args.checkpoint_model_suffix = None
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    else:
        try:
            args.checkpoint_model_suffix = max([int(re.search("\d+.ckpt", line)[0][:-5]) for line in
                                                glob.glob(os.path.join(args.model_path, "*[0-9].ckpt")) if
                                                "YC" not in line])
            args.start_epoch = args.checkpoint_model_suffix
        except Exception:
            print(f"{args.model_path} already exists but no checkpoint is found. Starting training from scratch")

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.model_path, "logs_recipe1M.log"), mode='a+'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Configure wandb
    if args.wandb_log:
        initialize_wandb(args, wandb, model_folder, "Recipe1M", wandb_id=args.wandb_id)

    logging.info(f"Run name: {model_folder}")
    logging.info(f"Args: {vars(args)}")
    if args.checkpoint_model_suffix is not None:
        logging.info(f"Resuming training from epoch {args.start_epoch}")

    train(args, eval_only=args.eval_only)
