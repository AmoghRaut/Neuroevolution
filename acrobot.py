import torch.nn as nn


class AcroBotAI(nn.Module):
        def __init__(self,actions):
            super().__init__()
            self.fc = nn.Sequential(
                        nn.Linear(6,64, bias=True),
                        nn.ReLU(),
                        nn.Linear(64,128, bias=True),
                        nn.ReLU(),
                        nn.Linear(128,actions, bias=True),
                        nn.Softmax(dim=1)
                        )

                
        def forward(self, inputs):
            x = self.fc(inputs)
            return x
