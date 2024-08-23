# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model. """


import math
import os
import warnings
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss

from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    ModelOutput
)
from torch.nn.init import xavier_uniform_
# peft specific
from peft.utils import ModulesToSaveWrapper

class CombinedSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        label_attentions (`torch.FloatTensor` of shape `(batch_size, num_labels, sequence_length)`):
            Attentions weights for the labels, used to compute the weighted average in the label-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    label_attentions: Optional[torch.FloatTensor] = None

class RobertaForCombinedSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model_mode = config.model_mode

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if "cls" in self.model_mode:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        elif "laat" in self.model_mode:
            self.first_linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.second_linear = nn.Linear(config.hidden_size, config.num_labels, bias=False)
            self.third_linear = nn.Linear(config.hidden_size, config.num_labels)
        else:
            raise ValueError(f"model_mode {self.model_mode} not recognized")

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        output_label_attentions=None,
    ):
        r"""
        input_ids (torch.LongTensor of shape (batch_size, num_chunks, chunk_size))
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_labels)`, `optional`):
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, num_chunks, chunk_size = input_ids.size()
        # print(f"batch_size: {batch_size}, num_chunks: {num_chunks}, chunk_size: {chunk_size}")
        # when running through roberta we change shape
        # print(f"input_ids view -1 .shape: {input_ids.view(-1, chunk_size).shape}")
        outputs = self.roberta(
            input_ids.view(-1, chunk_size),
            attention_mask=attention_mask.view(-1, chunk_size) if attention_mask is not None else None,
            token_type_ids=token_type_ids.view(-1, chunk_size) if token_type_ids is not None else None,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print(f"First outputs shape is: {outputs[0].shape}")
        if "cls" in self.model_mode:
            pooled_output = outputs[1].view(batch_size, num_chunks, -1)
            if self.model_mode == "cls-sum":
                pooled_output = pooled_output.sum(dim=1)
            elif self.model_mode == "cls-max":
                pooled_output = pooled_output.max(dim=1).values
            else:
                raise ValueError(f"model_mode {self.model_mode} not recognized")
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
        elif "laat" in self.model_mode:
            if self.model_mode == "laat":
                # print(f"Reshape outputs[0] to: {outputs[0].view(batch_size, num_chunks*chunk_size, -1).shape}")
                hidden_output = outputs[0].view(batch_size, num_chunks*chunk_size, -1)
            elif self.model_mode == "laat-split":
                hidden_output = outputs[0].view(batch_size*num_chunks, chunk_size, -1)
                
            # lets remove the pad token representations from here - otherwise they seem to influence the attn
            # esentially we just remove any tokens where the attn mask is 0 - i.e. the pad tokens
            hidden_output = hidden_output[attention_mask.view(batch_size, num_chunks*chunk_size) == 1].unsqueeze(0)
                
            # print(f"hidden_output shape is: {hidden_output.shape}")
            weights = torch.tanh(self.first_linear(hidden_output))
            # print(f"weights shape is: {weights.shape}")
            att_weights = self.second_linear(weights)
            # print(f"first att_weights shape is: {att_weights.shape}")
            # att_weights.masked_fill_((attention_mask.view(batch_size, -1, 1)==0), -math.inf)
            att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)
            # print(f"after softmax att_weights shape is: {att_weights.shape}")
            weighted_output = att_weights @ hidden_output
            # print(f"weighted_output shape is: {weighted_output.shape}")
            # print(f"The third linear layer has shape: {self.third_linear.weight.shape}")
            # print(f"The logits shape before summing is: {self.third_linear.weight.mul(weighted_output).shape}")
            #NOTE - janky fix for combining PEFT as it wraps the modules and means the format changes
            if isinstance(self.third_linear, ModulesToSaveWrapper):
                #NOTE - we actually want to used the modules_to_save part of the module
                logits = self.third_linear.modules_to_save.default.weight.mul(weighted_output).sum(dim=2).add(self.third_linear.modules_to_save.default.bias)
            else:
                logits = self.third_linear.weight.mul(weighted_output).sum(dim=2).add(self.third_linear.bias)
            # print(f"logits shape is: {logits.shape}")
            if self.model_mode == "laat-split":
                logits = logits.view(batch_size, num_chunks, -1).max(dim=1).values
        else:
            raise ValueError(f"model_mode {self.model_mode} not recognized")

        loss = None
        # print(f"logits shape is: {logits.shape}")
        # print(f"labels shape is: {labels.shape}")
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CombinedSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            label_attentions = att_weights if "laat" in self.model_mode else None,
        )
        

class LabelAttention(nn.Module):
    def __init__(self, input_size: int, projection_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self.third_linear = nn.Linear(input_size, num_classes)
        self._init_weights(mean=0.0, std=0.03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LAAT attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        weights = torch.tanh(self.first_linear(x))
        att_weights = self.second_linear(weights)
        att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)
        weighted_output = att_weights @ x
        return (
            self.third_linear.weight.mul(weighted_output)
            .sum(dim=2)
            .add(self.third_linear.bias)
        )

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        torch.nn.init.normal_(self.first_linear.weight, mean, std)
        torch.nn.init.normal_(self.second_linear.weight, mean, std)
        torch.nn.init.normal_(self.third_linear.weight, mean, std)


class CAMLAttention(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, num_classes)
        xavier_uniform_(self.first_linear.weight)
        self.second_linear = nn.Linear(input_size, num_classes)
        xavier_uniform_(self.second_linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """CAML attention mechanism

        Args:
            x (torch.Tensor): [batch_size, input_size, seq_len]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        x = torch.tanh(x)
        weights = torch.softmax(self.first_linear.weight.matmul(x), dim=2)
        weighted_output = weights @ x.transpose(1, 2)
        return (
            self.second_linear.weight.mul(weighted_output)
            .sum(2)
            .add(self.second_linear.bias)
        )
