import torch

#test
class AE(torch.nn.Module):
	def __init__(self, inSize, outSize):
		super().__init__()
		#self. insize = inSize
		#self.outSize = outSize

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
			#self.double()
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

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded
