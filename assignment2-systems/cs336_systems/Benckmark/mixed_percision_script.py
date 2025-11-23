import torch

# if torch.cuda.is_available():
#     torch.set_default_device("cuda")  # 之后新建张量默认都在GPU
# # 你的原始代码可直接复用（dtype保持一致）
# s = torch.tensor(0,dtype=torch.float32)
# for i in range(1000):
#     s += torch.tensor(0.01,dtype=torch.float32)
# print(s)
# s = torch.tensor(0,dtype=torch.float16)
# for i in range(1000):
#     s += torch.tensor(0.01,dtype=torch.float16)
# print(s)
# s = torch.tensor(0,dtype=torch.float32)
# for i in range(1000):
#     s += torch.tensor(0.01,dtype=torch.float16)
# print(s)
# s = torch.tensor(0,dtype=torch.float32)
# for i in range(1000):
#     x = torch.tensor(0.01,dtype=torch.float16)
#     s += x.type(torch.float32)
# print(s)


import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print("After fc1:", x.dtype)
        x = self.ln(x)
        print("After layer norm:", x.dtype)
        x = self.fc2(x)
        print("After fc2:", x.dtype)
        return x


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ToyModel(8, 4).to(device)
    x = torch.randn(2, 8, device=device)
    print(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = model(x)
    print("Final logits:", y.dtype)
    loss = y.sum()
    print("Loss dtype:", loss.dtype)

    loss.backward()
    for name, param in model.named_parameters():
        print(name, "grad dtype:", param.grad.dtype)

if __name__ == '__main__':
    main()