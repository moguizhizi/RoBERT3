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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import codecs
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import scipy.linalg

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
PathProject = os.path.split(rootPath)[0]
sys.path.append(rootPath)
sys.path.append(PathProject)
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from common.evaluators.bert_evaluator import BertEvaluator
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from common.trainers.bert_trainer import BertTrainer
from transformers.file_utils import PYTORCH_TRANSFORMERS_CACHE
from transformers import WarmupLinearSchedule
from transformers.modeling_bert import BertForSequenceClassification, BertConfig#, WEIGHTS_NAME, CONFIG_NAME
from transformers.tokenization_bert import BertTokenizer
from transformers.optimization import AdamW
from topics import topic_num_map
from args import get_args
from pytorch_pretrained_bert import BertModel
from datasets.bert_processors.aapd_processor import AAPDProcessor,MultiHeadedAttention,PositionwiseFeedForward,LayerNorm,SublayerConnection,EncoderLayer,EncoderLayer1
from common.constants import *
from preprocess_situation import evaluate_situation_zeroshot_TwpPhasePred
# import torch.optim as optimizer_wenpeng
import copy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class ClassifyModel(customizedModule):
    def __init__(self,pretrained_model_name_or_path,num_labels,Encoder1,is_lock = False):
        super(ClassifyModel,self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.Encoder = Encoder1
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768,num_labels)
        self.init_mBloSA()
        self.s2tSA = s2tSA(768)
        self.Attention = Attention(768,8)
        self.gru = torch.nn.GRU(768, 768, 1)
        if is_lock:
            for name, param in self.bert.named_parameters():
                if name.startswith('pooler'):
                    continue
                else:
                    param.requires_grad_(False)

    def init_mBloSA(self):
        self.g_W1 = self.customizedLinear(768, 768)
        self.g_W2 = self.customizedLinear(768, 768)
        self.g_b = nn.Parameter(torch.zeros(768))

        self.g_W1[0].bias.requires_grad = False
        self.g_W2[0].bias.requires_grad = False

        self.f_W1 = self.customizedLinear(768 * 2, 768, activation=nn.ReLU())
        self.f_W2 = self.customizedLinear(768 * 2, 768)

    def forward(self, input_ids,token_type_ids = None,attention_mask = None,label = None,):
        _,pooled = self.bert(input_ids,token_type_ids,attention_mask,output_all_encoded_layers = False)
        logits = self.dropout(pooled)
        logits = self.classifier(logits)
        return logits
def main():
    #Set default configuration in args.py
    args = get_args()
    dataset_map = {'AAPD': AAPDProcessor}

    output_modes = {"rte": "classification"}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')
    args.device = device
    args.n_gpu = n_gpu  # 1
    args.is_hierarchical = True
    args.num_labels = dataset_map[args.dataset].NUM_CLASSES  # 12
    args.is_multilabel = dataset_map[args.dataset].IS_MULTILABEL  # True
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not args.trained_model:
        save_path = os.path.join(args.save_path, dataset_map[args.dataset].NAME)
        os.makedirs(save_path, exist_ok=True)

    processor = dataset_map[args.dataset]()

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    MultiHeadedAttention1 = MultiHeadedAttention(16,768)
    PositionwiseFeedForward1 = PositionwiseFeedForward(768,3072)
    EncoderLayer_1 = EncoderLayer(768,MultiHeadedAttention1,PositionwiseFeedForward1,0.1)
    Encoder1 = Encoder(EncoderLayer_1, 2)
    pretrain_model_dir = '/home/ltf/code/data/bert-base-uncased/'
    model = ClassifyModel(pretrain_model_dir, num_labels=args.num_labels, Encoder1=Encoder1, is_lock=False)
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)

    if args.fp16:
        model.half()
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if not args.trained_model:
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install NVIDIA Apex for FP16 training")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.lr,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=0.01, correct_bias=False)
            scheduler = WarmupLinearSchedule(optimizer, t_total=num_train_optimization_steps,
                                             warmup_steps=args.warmup_proportion * num_train_optimization_steps)

        trainer = BertTrainer(model, optimizer, processor, scheduler, tokenizer, args)
        trainer.train()
        model = torch.load(trainer.snapshot_path)
    else:
        model = BertForSequenceClassification.from_pretrained(pretrain_model_dir, num_labels=args.num_labels)
        model_ = torch.load(args.trained_model, map_location=lambda storage, loc: storage)
        state = {}
        for key in model_.state_dict().keys():
            new_key = key.replace("module.", "")
            state[new_key] = model_.state_dict()[key]
        model.load_state_dict(state)
        model = model.to(device)

    evaluate_split(model, processor, tokenizer, args, split='dev')
    evaluate_split(model, processor, tokenizer, args, split='test')

if __name__ == "__main__":
    main()
