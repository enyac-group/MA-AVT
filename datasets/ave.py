import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
# from ipdb import set_trace
import pickle as pkl
import h5py

import soundfile as sf
import torchaudio

import torchvision
import glob
import random

### VGGSound
from scipy import signal
import soundfile as sf
###

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


import warnings
warnings.filterwarnings('ignore')


class AVE(Dataset):

	def __init__(self, opt, mode='train'):

		self.opt = opt
		self.audio_folder = os.path.join(opt.data_dir, 'AVE_Dataset', 'raw_audio')
		self.video_folder = os.path.join(opt.data_dir, 'AVE_Dataset', 'video_frames')

		with h5py.File(os.path.join(opt.data_dir, 'labels.h5'), 'r') as hf:
			self.labels = hf['avadataset'][:]

		if mode == 'train':
			with h5py.File(os.path.join(opt.data_dir, 'train_order.h5'), 'r') as hf:
				order = hf['order'][:]
		elif mode == 'val':
			with h5py.File(os.path.join(opt.data_dir, 'val_order.h5'), 'r') as hf:
				order = hf['order'][:]
		elif mode == 'test':
			with h5py.File(os.path.join(opt.data_dir, 'test_order.h5'), 'r') as hf:
				order = hf['order'][:]

		self.lis = order
		self.mode = mode
		self.raw_gt = pd.read_csv(os.path.join(opt.data_dir, "Annotations.txt"), sep="&", header=None)

		### ---> for audio in AST
		# self.norm_mean = -5.4450
		# self.norm_std = 2.9610
		### <----

		### ---> for audio in AST
		# norm_stats = {'audioset':[-4.2677393, 4.5689974], 
		# 'esc50':[-6.6268077, 5.358466], 
		# 'speechcommands':[-6.845978, 5.5654526]}
		# check ast/src/get_norm_stats.py
		# self.norm_mean = -4.2677393
		# self.norm_std = 4.5689974
		### <----

		### ---> yb calculate: AVE dataset
		if self.opt.vis_encoder_type == 'vit':
			self.norm_mean = -4.1426
			self.norm_std = 3.2001
		### <----
		
		elif self.opt.vis_encoder_type == 'swin':
			## ---> yb calculate: AVE dataset for 192
			self.norm_mean =  -4.984795570373535
			self.norm_std =  3.7079780101776123
			## <----
			
		if self.opt.vis_encoder_type == 'vit':
			self.my_normalize = Compose([
				Resize([224, 224], interpolation=Image.BICUBIC),
				Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
			])
		
		self.opt.audio_length = 1.0

		# elif self.opt.vis_encoder_type == 'swin':
		# 	self.my_normalize = Compose([
		# 		Resize([192,192], interpolation=Image.BICUBIC),
		# 		Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
		# 	])

	def _wav2fbank(self, filename, idx=None, return_details=False):		
		waveform, sr = torchaudio.load(filename)
		if return_details:
			output_waveform = waveform

		waveform_mean = waveform.mean()
		waveform = waveform - waveform.mean()

		if waveform.shape[1] > 16000*(self.opt.audio_length+0.1):
			sample_indx = np.linspace(0, waveform.shape[1] -16000*(self.opt.audio_length+0.1), num=10, dtype=int)
			waveform = waveform[:,sample_indx[idx]:sample_indx[idx]+int(16000*self.opt.audio_length)]

		if self.opt.vis_encoder_type == 'vit':
			fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=224, dither=0.0, frame_shift=4.4)
			# fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=512, dither=0.0, frame_shift=1)
		elif self.opt.vis_encoder_type == 'swin':
			fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=192, dither=0.0, frame_shift=5.2)

		########### ------> very important: audio normalized
		fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
		### <--------
		if self.opt.vis_encoder_type == 'vit':
			target_length = 224
			# target_length = int(1024 * (1/10)) ## for audioset: 10s
		elif self.opt.vis_encoder_type == 'swin':
			target_length = 192 ## yb: overwrite for swin

		n_frames = fbank.shape[0]
		p = target_length - n_frames

		# cut and pad
		if p > 0:
			m = torch.nn.ZeroPad2d((0, 0, 0, p))
			fbank = m(fbank)
		elif p < 0:
			fbank = fbank[0:target_length, :]

		return fbank

	def __len__(self):
		return len(self.lis)

	def __getitem__(self, idx):
		real_idx = self.lis[idx]
		file_name = self.raw_gt.iloc[real_idx][1]

		### ---> loading all audio frames
		total_audio = []
		for audio_sec in range(10):
			fbank = self._wav2fbank(self.audio_folder + '/' + file_name + '.wav', idx=audio_sec)
			total_audio.append(fbank)

		total_audio = torch.stack(total_audio)
		### <----

		### ---> video frame process 
		total_num_frames = len(glob.glob(self.video_folder+'/'+file_name+'/*.jpg'))
		sample_indx = np.linspace(1, total_num_frames , num=10, dtype=int)
		total_img = []
		for vis_idx in range(10):
			tmp_idx = sample_indx[vis_idx]
			tmp_img = torchvision.io.read_image(self.video_folder+'/'+file_name+'/'+ str("{:06d}".format(tmp_idx))+ '.jpg')/255
			tmp_img = self.my_normalize(tmp_img)
			total_img.append(tmp_img)
		total_img = torch.stack(total_img)
		### <---
		
		return {'audio_spec': total_audio, 
				'image': total_img,
				'target': self.labels[real_idx]}