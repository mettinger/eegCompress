import torch

def sizeToLayerList(sizeList, finalStripBool):
	layerList = []
	for i in range(0, len(sizeList) - 1):
		layerList.append(torch.nn.Linear(sizeList[i], sizeList[i+1]))
		layerList.append(torch.nn.ReLU())

	if finalStripBool:
		layerList.pop()
	return layerList



class AE(torch.nn.Module):
	def __init__(self, encoderSizeList, decoderSizeList):
		super().__init__()

		finalStripBool = encoderSizeList.pop()
		encoderLayerList = sizeToLayerList(encoderSizeList, finalStripBool)
		finalStripBool = decoderSizeList.pop()
		decoderLayerList = sizeToLayerList(decoderSizeList, finalStripBool)

		self.encoder = torch.nn.Sequential(*encoderLayerList)
		self.decoder = torch.nn.Sequential(*decoderLayerList)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded

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