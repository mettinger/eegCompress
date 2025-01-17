import torch
from torch.utils.data.dataset import Dataset
import numpy as np

def sizeToLayerList(encoderSizeList, 
					decoderSizeList, 
					encoderActivationList, 
					decoderActivationList):
	encoderLayerList = []
	decoderLayerList = []

	for i in range(0, len(encoderSizeList) - 1):
		thisLayer = torch.nn.Linear(encoderSizeList[i], encoderSizeList[i + 1])
		#torch.nn.init.xavier_uniform_(thisLayer.weight) 

		encoderLayerList.append(thisLayer)
		if encoderActivationList[i]:
			encoderLayerList.append(torch.nn.ReLU())

	#decoderSizeList = [encoderSizeList[-2]] + decoderSizeList
	for i in range(0, len(decoderSizeList) - 1):
		thisLayer = torch.nn.Linear(decoderSizeList[i], decoderSizeList[i + 1])
		#torch.nn.init.xavier_uniform_(thisLayer.weight) 

		decoderLayerList.append(thisLayer)
		if decoderActivationList[i]:
			decoderLayerList.append(torch.nn.ReLU())

	return encoderLayerList, decoderLayerList

class AE(torch.nn.Module):
	def __init__(self, encoderSizeList, decoderSizeList, encoderActivationList, decoderActivationList):
		super().__init__()

		encoderLayerList, decoderLayerList = sizeToLayerList(encoderSizeList, 
															decoderSizeList, 
															encoderActivationList, 
															decoderActivationList)

		self.encoder = torch.nn.Sequential(*encoderLayerList)
		self.decoder = torch.nn.Sequential(*decoderLayerList)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

class CustomDataset(Dataset):
	def __init__(self, eegNumpy, numSampleInput):
		self.eegNumpy = eegNumpy
		self.numSampleInput = numSampleInput
		self.nChannel, self.nSample = eegNumpy.shape

	def __len__(self):
		return self.nSample - self.numSampleInput
		
	def __getitem__(self, idx):
		image = np.reshape(self.eegNumpy[:,idx:idx + self.numSampleInput], (self.nChannel * self.numSampleInput,-1), order='F').transpose().astype('float32')
		return image, 0

