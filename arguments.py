#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
# from util import util
import torch

class ArgParser():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False

	def initialize(self):
		self.parser.add_argument('--id', default='',
							help="a name for identifying the model"),
		self.parser.add_argument(
			"--dataset", type=str, default='ave', help="type of dataset")
		self.parser.add_argument('--batch_size', type=int, default=16, metavar='N',
							help='input batch size for training (default: 16)')
		self.parser.add_argument('--epochs', type=int, default=40, metavar='N',
							help='number of epochs to train (default: 60)')
		self.parser.add_argument('--start_epoch', default=1, type=int,
							help='epochs to start training for')
		
		############# yb param ###########
		self.parser.add_argument(
			"--model", type=str, default='MMIL_Net', help="with model to use")
		self.parser.add_argument(
			"--mode", type=str, default='train', help="with mode to use")
		# self.parser.add_argument('--seed', type=int, default=1, metavar='S',
		# 					help='random seed (default: 1)')
		self.parser.add_argument('--log-interval', type=int, default=50, metavar='N',
							help='how many batches to wait before logging training status')
		# self.parser.add_argument(
		# 	"--model_save_dir", type=str, default='models/', help="model save dir")

		self.parser.add_argument(
			"--output_dir", type=str, default='outputs/', help="model save dir")
		self.parser.add_argument('--fps', type=int, default=1)

		self.parser.add_argument("--clip_norm", type=float, default=0, help="gradient clip norm")
		# self.parser.add_argument(
		# 	"--checkpoint", type=str, default='cvpr_best',
		# 	help="save model name")
		# self.parser.add_argument(
		# 	'--gpu', type=str, default='0,1,2,3,4,5,6,7', help='gpu device number')
		# self.parser.add_argument(
		# 	'--wandb', type=int, default='0', help='weight and bias setup')

		# self.parser.add_argument(
		# 	'--is_v_ori', type=int, default='0', help='original visual features')

		# self.parser.add_argument(
		# 	'--is_a_ori', type=int, default='0', help='original audio features')

		# self.parser.add_argument(
		# 	'--tsne', type=int, default='0', help='run tsne or not')
		# self.parser.add_argument(
		# 	'--early_stop', type=int, default='5', help='weight and bias setup')

		# self.parser.add_argument(
		# 	'--threshold', type=float, default=0.5, help='weight and bias setup')
		
		# self.parser.add_argument(
		# 	'--save_model', action="store_true"
		# )

		# self.parser.add_argument(
		# 	'--pretrained', action="store_true"
		# )

		# self.parser.add_argument(
		# 	"--tmp", type=float, default=0.5,
		# 	help="video dir")

		# self.parser.add_argument(
		# 	"--noisy_label", action="store_true", default=False, help="audio dir")

		# self.parser.add_argument(
		# 	"--smooth", type=float, default=1,
		# 	help="video dir")
		
		### yb param ##
		############# yb param ###########
		self.parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
							help='learning rate (default: 3e-4)')
		self.parser.add_argument('--lr_mlp', type=float, default=1e-4, metavar='LR',
							help='learning rate (default: 3e-4)')
		self.parser.add_argument('--lr_v', type=float, default=3e-4, metavar='LR',
							help='learning rate (default: 3e-4)')
		
		self.parser.add_argument('--occ_dim', type=int, default=64, metavar='LR',
							help='learning rate (default: 3e-4)')

		self.parser.add_argument('--init_epoch', type=int, default=5, metavar='LR',
							help='learning rate (default: 3e-4)')

		self.parser.add_argument('--lr_step',
							default=10, type=int,
							help='steps to drop LR in epochs')

		self.parser.add_argument(
			'--margin1', type=float, default=0.05, help='weight and bias setup')

		self.parser.add_argument(
			'--alpha', type=float, default=1, help='weight and bias setup')
		
		self.parser.add_argument(
			'--beta', type=float, default=1, help='weight and bias setup')
		
		self.parser.add_argument(
			'--delta', type=float, default=1, help='weight and bias setup')
		
		self.parser.add_argument(
			'--gamma', type=float, default=1, help='weight and bias setup')
		
		self.parser.add_argument(
			'--decay', type=float, default=0.1, help='decay rate')
		
		self.parser.add_argument(
			'--decay_epoch', type=float, default=10, help='decay rate')

		self.parser.add_argument(
			'--aug_type', type=str, default='vision', help='weight and bias setup')

		self.parser.add_argument(
			'--pos_pool', type=str, default='max', help='weight and bias setup')
		
		self.parser.add_argument(
			'--neg_pool', type=str, default='mean', help='weight and bias setup')
		
		self.parser.add_argument(
			'--exp', type=int, default=0, help='weight and bias setup')
		
		self.parser.add_argument(
			'--ybloss', type=int, default=1, help='decay rate')

		### for transformer ###
		self.parser.add_argument(
			'--num_layer', type=int, default=1, help='num layer')
		
		self.parser.add_argument(
			'--num_head', type=int, default=1, help='num layer')
		
		self.parser.add_argument(
			'--prob_drop', type=float, default=0.1, help='drop out')
		
		self.parser.add_argument(
			'--prob_drop_occ', type=float, default=0.1, help='drop out')
		
		self.parser.add_argument(
			'--forward_dim', type=int, default=512, help='drop out')

		self.parser.add_argument(
			'--epoch_remove', type=int, default=1, help='weight and bias setup')
		#######################

		self.parser.add_argument(
			'--audio_enc', type=int, default= 0, help='weight and bias setup')

		self.parser.add_argument(	
			'--num_remove', type=int, default= 4, help='num of instances removing')


		### for AV-ada ###
		self.parser.add_argument('--audio_folder', type=str, default="/data/yanbo/Dataset/AVE_Dataset/raw_audio", help="raw audio path")
		self.parser.add_argument('--video_folder', type=str, default="/data/yanbo/Dataset/AVE_Dataset/video_frames", help="video frame path")
		self.parser.add_argument('--audio_length', type=float, default= 1, help='audio length')
		self.parser.add_argument('--num_workers', type=int, default= 4, help='worker for dataloader')
		self.parser.add_argument('--model_name', type=str, default=None, help="for log")

		self.parser.add_argument('--qkv_fusion', type=int, default=1, help="qkv fusion")

		self.parser.add_argument('--adapter_kind', type=str, default='bottleneck', help="for log")

		self.parser.add_argument('--start_tune_layers', type=int, default=0, help="tune top k")

		self.parser.add_argument('--start_fusion_layers', type=int, default=0, help="tune top k")

		self.parser.add_argument('--Adapter_downsample', type=int, default=16, help="tune top k")


		self.parser.add_argument('--num_conv_group', type=int, default=2, help="group conv")

		self.parser.add_argument('--log_path', type=str, default='/home/tm36864/LAVISH/AVE/logs', help="for log")

		self.parser.add_argument('--is_audio_adapter_p1', type=int, default=0, help="TF audio adapter")
		self.parser.add_argument('--is_audio_adapter_p2', type=int, default=0, help="TF audio adapter")
		self.parser.add_argument('--is_audio_adapter_p3', type=int, default=0, help="TF audio adapter")

		self.parser.add_argument('--is_bn', type=int, default=0, help="TF audio adapter")
		self.parser.add_argument('--is_gate', type=int, default=0, help="TF audio adapter")
		self.parser.add_argument('--is_multimodal', type=int, default=1, help="TF audio adapter")
		self.parser.add_argument('--is_before_layernorm', type=int, default=1, help="TF audio adapter")
		self.parser.add_argument('--is_post_layernorm', type=int, default=1, help="TF audio adapter")

		self.parser.add_argument('--is_vit_ln', type=int, default=0, help="TF Vit")

		self.parser.add_argument('--is_fusion_before', type=int, default=0, help="TF Vit")


		### my parameters
		self.parser.add_argument('--print_freq', type=int, default=1, help="print/logging frequency")

		self.parser.add_argument('--num_tokens', type=int, default=32, help="num of MBT tokens")

		self.parser.add_argument('--vis_encoder_type', type=str, default="swin", help="type of visual backbone")

		self.parser.add_argument('--vit_type', type=str, default=None, help="type of transformer backbone")

		self.parser.add_argument('--exp_name', type=str, default="", help="name of the experiment")
		
		self.parser.add_argument('--n_vis_tokens', type=int, default=0, help="num of MBT tokens")
		self.parser.add_argument('--n_audio_tokens', type=int, default=0, help="num of MBT tokens")
		self.parser.add_argument('--n_shared_tokens', type=int, default=0, help="num of MBT tokens")
		self.parser.add_argument('--eval_epoch', type=int, default=1, help="eval epoch")

		self.parser.add_argument('--checkpoint_path', type=str, default='/home/tm36864/LAVISH/AVE/logs', help="for log")
		self.parser.add_argument(
			'--load_model', action="store_true"
		)
		self.parser.add_argument(
			'--r', type=float, default=0.8
		)
		self.parser.add_argument(
			'--loss2', action="store_true"
		)

		self.parser.add_argument(
			'--freeze', action="store_true"
		)


	def add_mgpu_arguments(self):
		parser = self.parser

		parser.add_argument('--num_class', type=int, default=28, help='num class')

		parser.add_argument('--data_dir', default='/home/tm36864/LAVISH/AVE/outputs/data', type=str,
							help='path to latest checkpoint (default: none)')

		parser.add_argument('--workers', type=int, default=4)
		parser.add_argument('--gpu', type=int, default=None)
		parser.add_argument('--ngpu', type=int, default=None)
		parser.add_argument('--world_size', type=int, default=1)
		parser.add_argument('--rank', type=int, default=0)
		parser.add_argument('--node', type=str, default='localhost')
		parser.add_argument('--port', type=int, default=12345)
		parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:3456')
		parser.add_argument('--dist-backend', default='nccl', type=str, help='')

		parser.add_argument('--multiprocessing_distributed', action='store_true')


		parser.add_argument("--lr_schedule", default='cte', help="learning rate schedule")
		parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
		parser.add_argument("--warmup_epochs", type=int, default=0, help="warmup epochs")
		# parser.add_argument("--seed", type=int, default=12345, help="random seed")
		parser.add_argument('--weight_decay', type=float, default=0, help='Weight Decay')


		# parser.add_argument('--dist-backend', default='nccl', type=str,
		# 					help='distributed backend')
		parser.add_argument('--seed', default=1234, type=int,
							help='seed for initializing training. ')
		parser.add_argument('--resume', default='', type=str, metavar='PATH',
							help='path to latest checkpoint (default: none)')
		parser.add_argument('--pretrained', dest='pretrained', action='store_true',
							help='use pre-trained model')
		parser.add_argument(
			"--noisy_label", action="store_true", default=False, help="audio dir")
		parser.add_argument('--num_vis', default=50, type=int,
							help='total number of visualizations. ')
		self.parser = parser

	def add_new_arguments(self):
		parser = self.parser

		parser.add_argument('--bg_label', type=int, default=28, help='backgroun class label')
		parser.add_argument(
			"--bg_cls", action="store_true", default=False, help="whether to use bg cls in training")
		parser.add_argument("--bg_prob", type=float, default=0.2, help="background cls probability")
		parser.add_argument(
			"--LSA", action="store_true", default=False, help="whether to use LSA in training")
		parser.add_argument(
			"--full_tune", action="store_true", default=False, help="whether to use LSA in training")

		parser.add_argument(
			"--unimodal_token", action="store_true", default=False, help="whether to use bg cls in training")
		parser.add_argument(
			"--multimodal_token", action="store_true", default=False, help="whether to use bg cls in training")
		parser.add_argument(
			"--grad_mod", action="store_true", default=False, help="whether to use bg cls in training")
		parser.add_argument(
			"--lavish_adapter", action="store_true", default=False, help="whether to use bg cls in training")

		parser.add_argument(
			"--layerwise_token", type=str, default=None, help="whether to use bg cls in training")
		parser.add_argument(
			"--layerwise_attn", type=str, default=None, help="whether to use bg cls in training")
		parser.add_argument(
			"--contrastive", type=str, default=None, help="whether to use bg cls in training")

		self.parser = parser


	def print_arguments(self, args):
		print("Input arguments:")
		for key, val in vars(args).items():
			print("{:16} {}".format(key, val))

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.add_mgpu_arguments()
		self.add_new_arguments()
		
		args = self.parser.parse_args()

		self.print_arguments(args)

		return args