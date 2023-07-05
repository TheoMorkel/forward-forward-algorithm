import torch


x = torch.rand(5, 3)
y = torch.sqrt(torch.sum(torch.pow(x-torch.mean(x),2), dim=1))/ torch.numel(x)
y_1 = x.pow(2).sum(dim=1)

print(x)
print(y)
print(y_1)