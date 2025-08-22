# -*- coding: utf-8 -*-
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence

from utils.utils import generate_causal_mask, generate_padding_mask


class ResidualProjection(nn.Module):
    def __init__(self, args):
        """Set the hyper-parameters and build the layers."""
        super(ResidualProjection, self).__init__()
        self.univl_dim = args.sentEnd_hiddens
        self.hidden_dim = args.recipe_inDim
        self.residual_enc = nn.Sequential(nn.Linear(self.univl_dim, self.univl_dim, bias=True),
                                          nn.ReLU(),
                                          nn.Linear(self.univl_dim, self.univl_dim, bias=True),
                                          nn.ReLU(),
                                          nn.Linear(self.univl_dim, self.univl_dim, bias=True))
        self.proj = nn.Linear(self.univl_dim, self.univl_dim, bias=False)
        self.proj_enc = nn.Linear(self.univl_dim, self.hidden_dim, bias=False)

    def forward(self, feats):
        features = self.residual_enc(feats)
        features += self.proj(feats)
        features = self.proj_enc(features)
        return features


class IngredientEncoder(nn.Module):
    def __init__(self, args):
        """Set the hyper-parameters and build the layers."""
        super(IngredientEncoder, self).__init__()
        self.ing_prj_dim = args.recipe_inDim
        self.ingredient_dim = args.ingredient_dim
        self.linear = nn.Linear(self.ingredient_dim, self.ing_prj_dim)
        self.bn = nn.BatchNorm1d(self.ing_prj_dim, momentum=0.01)

    def forward(self, feats):
        # feats --> [N, Nv]
        features = self.linear(feats)  # [N, d]
        features = self.bn(features)  # [N, 1024]
        return features


class PositionalEncoding(nn.Module):
    r"""Taken from Pytorch: Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=6000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        if not self.batch_first:
            pe = pe.transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim] or [batch size, sequence length, embed dim] if  batch_first
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ContextEncoder(nn.Module):
    def __init__(self, args):
        """Set the hyper-parameters and build the layers."""
        super(ContextEncoder, self).__init__()
        self.recipe_inDim = args.recipe_inDim
        self.recipe_nlayers = args.recipe_nlayers
        self.recipe_nheads = args.recipe_nheads

        encoder_layer = nn.TransformerEncoderLayer(self.recipe_inDim, self.recipe_nheads, dim_feedforward=1024,
                                                   dropout=0.1, activation="relu", batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.recipe_nlayers)
        self.pos_encoder = PositionalEncoding(self.recipe_inDim, dropout=0.1, batch_first=True)

    def forward(self, steps_feats, rec_lens, input_mask: torch.FloatTensor = None,
                padding_mask: torch.BoolTensor = None):
        # len(rec_lens) -> N
        assert steps_feats.shape[1] == rec_lens[0]
        steps_feats = self.pos_encoder(steps_feats)
        out = self.transformer_encoder(steps_feats, input_mask, padding_mask)  # [N, rec_lens, 1024]
        out = out[~padding_mask]
        return out  # [sum(rec_lens), 1024]

    def sample_next_step(self, ingredients, instructions):
        # ingredients --> [1,1024]
        # instructions --> [rec_len,1024]
        ingredients = ingredients.view(1, -1)  # [1,1024]
        ingredients = ingredients.unsqueeze(1)  # [1, 1, 1024] - [batch_size, 1, 1024]
        instructions = instructions.unsqueeze(0)
        recipes_v = torch.cat((ingredients, instructions), 1)[:, :-1]  # [1, rec_len, 1024]
        assert len(recipes_v.shape) == 3
        causal_mask = generate_causal_mask(recipes_v.shape[1], recipes_v.device)
        padding_mask = generate_padding_mask([recipes_v.shape[1]], recipes_v.device)

        sampled_instructions = self.transformer_encoder(recipes_v, causal_mask, padding_mask)
        # [N, rec_lens, 1024]  # [1, rec_len, 1024] -> (batch_size, rec_len, feature_size)
        return sampled_instructions


class CVAEBlock(nn.Module):
    def __init__(self, args):
        """Set the hyper-parameters and build the layers."""
        super(CVAEBlock, self).__init__()
        self.input_dim = args.recipe_inDim
        self.latent_dim = args.cvae_latent_dim
        self.hidden_dim = args.cvae_hidden_dim
        self.n_hidden_layers = args.cvae_n_hidden_layers
        self.cvae_concat_next = args.cvae_concat_next
        self.learnable_prior = args.learnable_prior

        # Create prior network
        prior_network = [nn.Linear(self.input_dim, self.hidden_dim, bias=True), nn.ReLU()]
        for i in range(self.n_hidden_layers):
            prior_network.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=True))
            prior_network.append(nn.ReLU())
        prior_network.append(nn.Linear(self.hidden_dim, self.latent_dim * 2, bias=True))
        self.prior_network = nn.Sequential(*prior_network)

        # Create posterior network
        if self.cvae_concat_next:
            posterior_network = [nn.Linear(self.input_dim * 2, self.hidden_dim, bias=True), nn.ReLU()]
        else:
            posterior_network = [nn.Linear(self.input_dim, self.hidden_dim, bias=True), nn.ReLU()]
        for i in range(self.n_hidden_layers):
            posterior_network.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=True))
            posterior_network.append(nn.ReLU())
        posterior_network.append(nn.Linear(self.hidden_dim, self.latent_dim * 2, bias=True))
        self.posterior_network = nn.Sequential(*posterior_network)

        # Create instruction embedding reconstruction network
        instruct_embed_decoder = [nn.Linear(self.latent_dim + self.input_dim, self.hidden_dim, bias=True), nn.ReLU()]
        for i in range(self.n_hidden_layers):
            instruct_embed_decoder.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=True))
            instruct_embed_decoder.append(nn.ReLU())
        instruct_embed_decoder.append(nn.Linear(self.hidden_dim, self.input_dim, bias=True))
        self.instruct_embed_decoder = nn.Sequential(*instruct_embed_decoder)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu_p, logvar_p, mu_q, logvar_q):
        b, _ = mu_p.shape
        kl = 0.5 * (logvar_p - logvar_q + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp() - 1)
        return kl.sum() / b

    def run_prior(self, c):
        assert len(c.shape) == 2
        if self.learnable_prior:
            output_p = self.prior_network(c)
            mu_p, logvar_p = output_p[:, :self.latent_dim], output_p[:, self.latent_dim:]
        else:
            mu_p = torch.zeros(1, 1, device=c.device).expand(c.shape[0], self.latent_dim)
            logvar_p = torch.zeros(1, 1, device=c.device).expand(c.shape[0], self.latent_dim)
        return mu_p, logvar_p

    def run_posterior(self, c_x):
        output_q = self.posterior_network(c_x)
        mu_q, logvar_q = output_q[:, :self.latent_dim], output_q[:, self.latent_dim:]
        return mu_q, logvar_q

    def sample_z(self, mu, logvar):
        return self._reparameterize(mu, logvar)

    def decode_z(self, z, c):
        output = self.instruct_embed_decoder(torch.cat([z, c], dim=1))
        return output

    def forward(self, c, n_samples=1, use_mean=False):
        mu_p, logvar_p = self.run_prior(c)
        outputs = []
        for iter in range(n_samples):
            if iter == 0 and use_mean:
                z = mu_p
            else:
                z = self.sample_z(mu_p, logvar_p)
            outputs.append(self.decode_z(z, c).unsqueeze(1))
        if n_samples > 1:
            outputs = torch.cat(outputs, dim=1)
        else:
            outputs = outputs[0].squeeze(1)
        return outputs

    def forward_train(self, recipe_enc, encoded_steps=None, rec_lens=None):
        c = recipe_enc
        if self.cvae_concat_next:
            assert encoded_steps is not None
            c_x = torch.cat([recipe_enc, encoded_steps], dim=1)
        else:
            assert rec_lens is not None
            c_x = torch.cat([recipe_enc[1:], recipe_enc[:1]], dim=0)
        mu_p, logvar_p = self.run_prior(c)
        mu_q, logvar_q = self.run_posterior(c_x)
        z = self.sample_z(mu_q, logvar_q)
        output = self.decode_z(z, c)
        kl_div = self.kl_divergence(mu_p, logvar_p, mu_q, logvar_q)
        return output, kl_div


class WordEmbedding(nn.Module):
    def __init__(self, args):
        """Set the hyper-parameters and build the layers."""
        super(WordEmbedding, self).__init__()
        self.word_emb_dim = args.word_dim  # 256
        self.vocab_len = args.vocab_len  # 30171
        self.embed = nn.Embedding(self.vocab_len, self.word_emb_dim)

    def forward(self, sent_words):
        # sent_words --> [Nb, Ns] -> [total num sent, max sent len.]
        return self.embed(sent_words)  # [Nb, Ns, 256]


class InstructionDecoder(nn.Module):
    def __init__(self, args):
        """Set the hyper-parameters and build the layers."""
        super(InstructionDecoder, self).__init__()
        self.sentDec_inDim = args.recipe_inDim
        self.word_dim = args.word_dim
        self.sentDec_hiddens = args.sentDec_hiddens
        self.vocab_len = args.vocab_len
        self.sentDec_nlayers = args.sentDec_nlayers

        self.lstm = nn.LSTM(self.word_dim, self.sentDec_hiddens, self.sentDec_nlayers, batch_first=True)
        self.linear = nn.Linear(self.sentDec_hiddens, self.vocab_len)

        self.linear_project = nn.Linear(self.sentDec_inDim, self.word_dim)

    def forward(self, recipe_enc, word_embs, sent_lens):
        """Decode sentence feature vectors and generates sentences."""
        # recipe_enc --> [Nb, 1024]
        # word_embs  --> [Nb, Ns, 256]
        # len(sent_lens)  --> Nb
        sent_lens = sent_lens.cpu()

        features = self.linear_project(recipe_enc)  # [Nb, 256]
        word_embs = torch.cat((features.unsqueeze(1), word_embs), 1)  # torch.Size([Nb, Ns + 1, 256])
        packed = pack_padded_sequence(word_embs, sent_lens, batch_first=True, enforce_sorted=False)
        # [0] -> [sum(sent_lens), 256]   [1] -> [sent_lens[0]]

        out, _ = self.lstm(packed)  # [0] -> [sum(sent_lens), 512]   [1] -> [sent_lens[0]]
        outputs = self.linear(out[0])  # [sum(sent_lens), Nw] -- Nw = number of words in the vocabulary
        return outputs

    def _sample(self, sampling_func, recipe_enc, embed_words, states=None, max_seq_length=35):
        # recipe_enc --> [Nb, 1024] or [Nb, Nz, 1024]
        sampled_ids = []
        original_recipe_shape = None
        if len(recipe_enc.shape) == 3:
            original_recipe_shape = recipe_enc.shape
            recipe_enc = recipe_enc.view(-1, original_recipe_shape[2])
        features = self.linear_project(recipe_enc)  # [Nb, 256]
        inputs = features.unsqueeze(1)  # [Nb, 1, 256]
        for i in range(max_seq_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: [Nb, 1, 512]

            outputs = self.linear(hiddens.squeeze(1))  # [Nb, Nw] -- Nw = number of words in the vocabulary

            predicted = sampling_func(outputs)  # Nb
            sampled_ids.append(predicted)

            inputs = embed_words(predicted)  # [Nb, 256]
            inputs = inputs.unsqueeze(1)  # [Nb, 1,256]

        sampled_ids = torch.stack(sampled_ids, 1)  # [Nb, max_seq_length]
        if original_recipe_shape is not None:
            sampled_ids = sampled_ids.view(original_recipe_shape[0], original_recipe_shape[1], -1)
        return sampled_ids

    def greedy_sample(self, recipe_enc: torch.Tensor, embed_words, states=None, max_seq_length=35):
        # recipe_enc --> [Nb, 1024] or [Nb, Nz, 1024]
        sampling_func = lambda x: torch.argmax(x, dim=-1)
        sampled_ids = self._sample(sampling_func, recipe_enc, embed_words, states=states, max_seq_length=max_seq_length)
        return sampled_ids

    def nucleus_sample(self, recipe_enc: torch.Tensor, embed_words, states=None, max_seq_length=35, top_p=0.9):
        # recipe_enc --> [Nb, 1024] or [Nb, Nz, 1024]
        # sampling_func = lambda x: torch.argmax(x, dim=-1)
        def sampling_func(logits):
            # convert logits to probabilities using softmax
            probs = F.softmax(logits, dim=-1)

            # sort the probabilities in descending order
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

            # compute the cumulative probabilities
            cum_probs = torch.cumsum(sorted_probs, dim=-1)

            # find the index of the last probability below the threshold p
            sorted_mask = (cum_probs < top_p)

            # ensure that at least one word is selected
            sorted_mask[:, 0] = True
            last_index = torch.sum(sorted_mask, dim=-1)
            max_probs = sorted_probs[torch.arange(sorted_probs.shape[0]), last_index]
            probs[probs <= max_probs.unsqueeze(-1)] = 0
            selected_words = torch.multinomial(probs, num_samples=1).squeeze()

            return selected_words

        sampled_ids = self._sample(sampling_func, recipe_enc, embed_words, states=states, max_seq_length=max_seq_length)
        return sampled_ids


class GEPSAN(nn.Module):
    def __init__(self, args):
        super(GEPSAN, self).__init__()
        # Single-Modality Encoder
        self.text_encoder = ResidualProjection(args)
        self.visual_encoder = ResidualProjection(args)

        # Recipe Encoder
        self.ingredient_encoder = IngredientEncoder(args)
        self.context_encoder = ContextEncoder(args)
        self.cvae_block = CVAEBlock(args)

        # Instruction Decoder
        self.word_embedding = WordEmbedding(args)
        self.instruction_decoder = InstructionDecoder(args)

    def copy_textual_encoder_to_visual(self):
        self.visual_encoder.load_state_dict(self.text_encoder.state_dict())

    def freeze_modality_encoder(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    def freeze_recipe_encoder(self):
        for param in self.ingredient_encoder.parameters():
            param.requires_grad = False
        for param in self.context_encoder.parameters():
            param.requires_grad = False
        for param in self.cvae_block.parameters():
            param.requires_grad = False

    def freeze_instruction_decoder(self):
        for param in self.word_embedding.parameters():
            param.requires_grad = False
        for param in self.instruction_decoder.parameters():
            param.requires_grad = False

    def _get_context_encoder_output(self, ingredients_ids: torch.Tensor, univl_feats: torch.Tensor,
                                    rec_lens: list, visual_modality=False):
        """
        Get the output of the context encoder
        :param ingredients_ids: A one hot tensor of ingredients of size Number of Recipes x Size of Ingredients
            Vocabulary
        :param rec_lens: A list of all the recipes length (recipes are required to be sorted from longest to shortest)
        :param univl_feats: The univl embeddings of all the steps (either text or video segment) of size Number of
            Instructions x embedding dim
        :param visual_modality: Whether the input_v are for the text or visual modality, and hence whether to use
            the text_encoder or visual_encoder
        """
        device = univl_feats.device
        """ 1. Project Sentence Embeddings obtained from UniVL """
        if visual_modality:
            input_instr_embeds = self.visual_encoder(univl_feats)
        else:
            input_instr_embeds = self.text_encoder(univl_feats)

        """ 2. Encode ingredient """
        ingredients_feats = self.ingredient_encoder(ingredients_ids).unsqueeze(1)  # [N, 1, d]

        """ 3. Apply Context Encoder """
        # Prepare input to Context Encoder
        instr_embeds_by_recipe = torch.split(input_instr_embeds, rec_lens, dim=0)
        instr_embeds_padded = pack_sequence(instr_embeds_by_recipe, enforce_sorted=True)
        instr_embeds_padded = pad_packed_sequence(instr_embeds_padded, batch_first=True)[0]  # [N, rec_lens[0], d]

        # Concatenate the ingredients embeddings and remove the last step from the recipe (effectively shifting the
        # instructions by one)
        instr_embeds_padded_shifted = torch.cat((ingredients_feats, instr_embeds_padded[:, :-1]),
                                                1)  # [N, rec_lens[0], d]

        # prepare attention and padding masks
        padding_mask = generate_padding_mask(rec_lens, device)
        input_mask = generate_causal_mask(rec_lens[0], device)

        # Get the output of the context encoder
        context_vectors = self.context_encoder(instr_embeds_padded_shifted, rec_lens, input_mask=input_mask,
                                               padding_mask=padding_mask)
        return input_instr_embeds, context_vectors

    def forward(self, ingredients_ids: torch.Tensor, univl_feats: torch.Tensor, rec_lens: list, visual_modality=False):
        """
        :param ingredients_ids: A one hot tensor of ingredients of size Number of Recipes x Size of Ingredients
            Vocabulary
        :param rec_lens: A list of all the recipes length (recipes are required to be sorted from longest to shortest)
        :param univl_feats: The univl embeddings of all the steps (either text or video segment) of size Number of
            Instructions x embedding dim
        :param visual_modality: Whether the input_v are for the text or visual modality, and hence whether to use
            the text_encoder or visual_encoder
        """
        input_instr_embeds, context_vectors = self._get_context_encoder_output(
            ingredients_ids=ingredients_ids, univl_feats=univl_feats, rec_lens=rec_lens,
            visual_modality=visual_modality)

        generated_instr_embeds, kl_loss = self.cvae_block.forward_train(context_vectors, input_instr_embeds, rec_lens)
        assert generated_instr_embeds.shape == input_instr_embeds.shape
        return generated_instr_embeds, input_instr_embeds, kl_loss

    def decode_embeddings(self, instr_embeds: torch.Tensor, target_instr_ids: torch.Tensor, instr_lens: list):
        """
        Decode the instruction embeddings, with the target instruction provided for teacher forcing (use this function
            for training)
        :param instr_embeds: A tensor of the embeddings of the instructions to be generated
        :param target_instr_ids: A padded tensor of the target instruction words indices of size Number of
            Instructions x Maximum Instruction Length
        :param instr_lens: A list of all the number of words per instruction
        """
        word_embs = self.word_embedding(target_instr_ids)  # [Nb, Ns, 256]
        generated_instr = self.instruction_decoder(instr_embeds, word_embs, instr_lens)
        return generated_instr

    def decode_embeddings_greedy(self, instr_embeds: torch.Tensor):
        """
        Decode the instruction embeddings gready sampling
        :param instr_embeds: A tensor of the embeddings of the instructions to be generated
        """
        generated_instr = self.instruction_decoder.greedy_sample(instr_embeds, self.word_embedding)
        return generated_instr

    def decode_embeddings_nucleus(self, instr_embeds: torch.Tensor):
        """
        Decode the instruction embeddings using nucleus sampling
        :param instr_embeds: A tensor of the embeddings of the instructions to be generated
        """
        generated_instr = self.instruction_decoder.nucleus_sample(instr_embeds, self.word_embedding)
        return generated_instr

    def generate(self, ingredients_ids: torch.Tensor, univl_feats: torch.Tensor, rec_lens: list, visual_modality=False,
                 n_samples=1, nucleus_instruction_decoding=False):
        """
        :param ingredients_ids: A one hot tensor of ingredients of size Number of Recipes x Size of Ingredients
            Vocabulary
        :param rec_lens: A list of all the recipes length (recipes are required to be sorted from longest to shortest)
        :param univl_feats: The univl embeddings of all the steps (either text or video segment) of size Number of
            Instructions x embedding dim
        :param visual_modality: Whether the input_v are for the text or visual modality, and hence whether to use
            the text_encoder or visual_encoder
        :param n_samples: number of instruction embeddings to sample from the CVAE block
        :param nucleus_instruction_decoding: Use nucleus sampling for decoding the instruction embeddings into
            instructions
        """
        input_instr_embeds, context_vectors = self._get_context_encoder_output(
            ingredients_ids=ingredients_ids, univl_feats=univl_feats, rec_lens=rec_lens,
            visual_modality=visual_modality)

        generated_instr_embeds_mean = self.cvae_block(context_vectors, n_samples=1, use_mean=True)
        generated_instr_embeds_sampled = self.cvae_block(context_vectors, n_samples=n_samples, use_mean=False)
        assert generated_instr_embeds_mean.shape == input_instr_embeds.shape

        if nucleus_instruction_decoding:
            generated_instr_mean = self.instruction_decoder.nucleus_sample(generated_instr_embeds_mean,
                                                                           self.word_embedding)
            generated_instr_sampled = self.instruction_decoder.nucleus_sample(generated_instr_embeds_sampled,
                                                                              self.word_embedding)
        else:
            generated_instr_mean = self.instruction_decoder.greedy_sample(generated_instr_embeds_mean,
                                                                          self.word_embedding)
            generated_instr_sampled = self.instruction_decoder.greedy_sample(generated_instr_embeds_sampled,
                                                                             self.word_embedding)
        return generated_instr_mean, generated_instr_sampled, \
               (generated_instr_embeds_mean, generated_instr_embeds_sampled)

