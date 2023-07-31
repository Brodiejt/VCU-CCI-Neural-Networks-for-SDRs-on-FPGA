from pytorch_nndct.apis import Inspector
import torch
import torch.nn as nn


model_path = 'model.pth'

class PimaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 12)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.act_output = nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x
 
model = PimaClassifier()

target = "DPUCZDX8G_ISA1_B4096"
inspector = Inspector(target)

dummy_input = torch.randn(1, 8, 12)
inspector.inspect(model, (dummy_input,), device=torch.device("cpu"),
                  output_dir="inspection_output_directory")