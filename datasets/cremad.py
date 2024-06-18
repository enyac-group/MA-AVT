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

class CremadDataset(Dataset):

	def __init__(self, args, mode='train', return_audio=False):
		self.args = args
		self.image = []
		self.audio = []
		self.label = []
		self.mode = mode

		self.bg_cls = args.bg_cls
		self.bg_label = args.bg_label
		self.bg_prob = args.bg_prob

		self.data_root = args.data_dir
		class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

		self.idx_to_class = {0:'NEU', 1:'HAP', 2:'SAD', 3:'FEA', 4:'DIS', 5:'ANG' }

		self.visual_feature_path = os.path.join(args.data_dir, 'frames')
		self.audio_feature_path = os.path.join(args.data_dir, 'AudioWAV')

		self.train_csv = os.path.join(self.data_root, 'Annotations', 'train.csv')
		self.test_csv = os.path.join(self.data_root, 'Annotations', 'test.csv')
		self.return_audio = return_audio

		if mode == 'train':
			csv_file = self.train_csv
		else:
			csv_file = self.test_csv

		with open(csv_file, encoding='UTF-8-sig') as f2:
			csv_reader = csv.reader(f2)
			for item in csv_reader:
				audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
				visual_path = os.path.join(self.visual_feature_path, 'Image-01-FPS', item[0])

				if os.path.exists(audio_path) and os.path.exists(visual_path):
					self.image.append(visual_path)
					self.audio.append(audio_path)
					self.label.append(class_dict[item[1]])
				else:
					continue


	def __len__(self):
		return len(self.image)

	def __getitem__(self, idx):

		# audio
		samples, rate = librosa.load(self.audio[idx], sr=22050)

		resamples = np.tile(samples, 3)[:22050*3]
		resamples[resamples > 1.] = 1.
		resamples[resamples < -1.] = -1.

		if self.return_audio:
			output_waveform = resamples

		# spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
		spectrogram = librosa.stft(resamples, n_fft=447, hop_length=296)
		spectrogram = np.log(np.abs(spectrogram) + 1e-7)

		if self.return_audio:
			output_spectrogram = spectrogram

		mean = np.mean(spectrogram)
		std = np.std(spectrogram)
		spectrogram = np.divide(spectrogram - mean, std + 1e-9)

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

		if self.bg_cls and self.mode == "train":
			if random.random() < self.bg_prob:
				new_label = label

				while new_label != label: 
					idx = torch.randint(range(len(self)))
					new_label = self.label[idx]
				
				label = self.bg_label

		image_samples = os.listdir(self.image[idx])
		
		if len(image_samples) > 1:
			select_index = np.random.choice((1, len(image_samples)), size=self.args.fps, replace=False)
			select_index.sort()
		else:
			select_index = [0]

		images = torch.zeros((self.args.fps, 3, 224, 224))
		for i in range(self.args.fps):
			img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
			img = transform(img)
			images[i] = img

		if self.return_audio:
			return {'audio_spec': torch.tensor(spectrogram).unsqueeze(0).float(), 
						'image': torch.tensor(images).float(), 
						'target': torch.tensor(label).unsqueeze(0),
						'waveform': torch.tensor(output_waveform),
						'spectrogram': torch.tensor(output_spectrogram)}


		return {'audio_spec': torch.tensor(spectrogram).unsqueeze(0).float(), 
					'image': torch.tensor(images).float(), 
					'target': torch.tensor(label).unsqueeze(0)}
