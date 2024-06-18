import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import random

class VGGSound(Dataset):

	def __init__(self, args, mode='train', return_audio=False):
		self.args = args
		self.mode = mode
		train_video_data = []
		train_audio_data = []
		test_video_data  = []
		test_audio_data  = []
		train_label = []
		test_label  = []
		train_class = []
		test_class  = []

		self.bg_cls = args.bg_cls
		self.bg_label = args.bg_label
		self.bg_prob = args.bg_prob
		self.return_audio = return_audio

		with open(os.path.join(args.data_dir, "vggsound.csv")) as f:
			csv_reader = csv.reader(f)

			for item in csv_reader:
				if item[3] == 'train':
					video_dir = os.path.join(args.data_dir, 'frames/train', 'Image-01-FPS', f"{item[0]}_{int(item[1]):06d}.mp4")
					audio_dir = os.path.join(args.data_dir, 'audio/train', f"{item[0]}_{int(item[1]):06d}.wav")

					if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir))>3 :
						train_video_data.append(video_dir)
						train_audio_data.append(audio_dir)
						if item[2] not in train_class: train_class.append(item[2])
						train_label.append(item[2])

				if item[3] == 'test':
					video_dir = os.path.join(args.data_dir, 'frames/test', 'Image-01-FPS', f"{item[0]}_{int(item[1]):06d}.mp4")
					audio_dir = os.path.join(args.data_dir, 'audio/test', f"{item[0]}_{int(item[1]):06d}.wav")

					if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir))>3:
						test_video_data.append(video_dir)
						test_audio_data.append(audio_dir)
						if item[2] not in test_class: test_class.append(item[2])
						test_label.append(item[2])

		train_class = sorted(train_class)
		test_class = sorted(test_class)

		assert len(train_class) == len(test_class)
		self.classes = train_class

		class_dict = dict(zip(self.classes, range(len(self.classes))))
		self.idx_to_class = {}
		for cls_name, idx in class_dict.items():
			self.idx_to_class[idx] = cls_name

		if mode == 'train':
			self.video = train_video_data
			self.audio = train_audio_data
			self.label = [class_dict[train_label[idx]] for idx in range(len(train_label))]
		else:
			self.video = test_video_data
			self.audio = test_audio_data
			self.label = [class_dict[test_label[idx]] for idx in range(len(test_label))]

	def __len__(self):
		return len(self.video)

	def __getitem__(self, idx):
		sample, rate = librosa.load(self.audio[idx], sr=16000, mono=True)

		while len(sample)/rate < 10.:
			sample = np.tile(sample, 2)

		if self.mode == "train":
			start_point = random.randint(a=0, b=rate*5)
		else:
			start_point = 3 * rate

		new_sample = sample[start_point:start_point+rate*5]
		new_sample[new_sample > 1.] = 1.
		new_sample[new_sample < -1.] = -1.

		if self.return_audio:
			output_waveform = new_sample

		spectrogram = librosa.stft(new_sample, n_fft=446, hop_length=358)
		
		if self.return_audio:
			output_spectrogram = spectrogram 

		spectrogram = np.log(np.abs(spectrogram) + 1e-7)

		mean = np.mean(spectrogram)
		std = np.std(spectrogram)
		spectrogram = np.divide(spectrogram-mean, std+1e-9)

		if self.mode == 'train':
			transform = transforms.Compose([
				transforms.RandomResizedCrop(224),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])
		else:
			transform = transforms.Compose([
				transforms.Resize(size=(224, 224)),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])


		# Visual
		label = self.label[idx]

		if self.bg_cls and self.mode == 'train':
			value = random.random() 
			if value < self.bg_prob:
				new_label = label

				while new_label != label: 
					idx = torch.randint(range(len(self)))
					new_label = self.label[idx]
				
				label = self.bg_label

		image_samples = sorted(os.listdir(self.video[idx]))
		
		try:
			select_idx = int(start_point // rate + 2.5)
			img = Image.open(os.path.join(self.video[idx], image_samples[select_idx])).convert('RGB')
		except:
			select_idx = np.random.choice(len(image_samples), size=(1,), replace=False)[0]
			img = Image.open(os.path.join(self.video[idx], image_samples[select_idx])).convert('RGB')
			
		img = transform(img)

		if self.return_audio:
			return {'audio_spec': torch.tensor(spectrogram).unsqueeze(0).float(), 
						'image': torch.tensor(img).unsqueeze(0).float(), 
						'target': torch.tensor(label).unsqueeze(0),
						'waveform': torch.tensor(output_waveform),
						'spectrogram': torch.tensor(output_spectrogram)}

		return {'audio_spec': torch.tensor(spectrogram).unsqueeze(0).float(), 
					'image': torch.tensor(img).unsqueeze(0).float(), 
					'target': torch.tensor(label).unsqueeze(0)}