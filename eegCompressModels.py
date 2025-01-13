import torch
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

class CustomDataset(torch.utils.data.dataset.Dataset):
	def __init__(self, eegNumpy, numSampleInput):
		self.eegNumpy = eegNumpy
		self.numSampleInput = numSampleInput
		self.nChannel, self.nSample = eegNumpy.shape

	def __len__(self):
		return self.nSample - self.numSampleInput
		
	def __getitem__(self, idx):
		image = np.reshape(self.eegNumpy[:,idx:idx + self.numSampleInput], (self.nChannel * self.numSampleInput,-1), order='F').transpose().astype('float32')
		return image, 0



'''
def sizeToLayerList(sizeList, finalStripBool):
	layerList = []
	for i in range(0, len(sizeList) - 1):
		thisLayer = torch.nn.Linear(sizeList[i], sizeList[i+1])
		#torch.nn.init.xavier_uniform_(thisLayer.weight) 

		layerList.append(thisLayer)
		layerList.append(torch.nn.ReLU())

	if finalStripBool:
		layerList.pop()
	return layerList
'''

'''
self.encoder = torch.nn.Sequential(
			torch.nn.Linear(inSize, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 36),
			torch.nn.ReLU(),
			torch.nn.Linear(36, 18),
			torch.nn.ReLU(),
			torch.nn.Linear(18, outSize)
		)
		
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(outSize, 18),
			torch.nn.ReLU(),
			torch.nn.Linear(18, 36),
			torch.nn.ReLU(),
			torch.nn.Linear(36, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, inSize),
			torch.nn.Sigmoid()
		)
'''