import torch


class SumSquared:
    def __init__(self):
        super().__init__()
        pass

    def __call__(self, x: torch.Tensor):
        return x.pow(2).sum(dim=1)

    def __str__(self) -> str:
        return "SumSquared"

class Sum:
    def __init__(self):
        super().__init__()
        pass

    def __call__(self, x: torch.Tensor):
        return x.sum(dim=1)

    def __str__(self) -> str:
        return "Sum"


class RootMeanSquare:
    def __init__(self):
        super().__init__()
        pass

    def __call__(self, x: torch.Tensor):
        return torch.sqrt(torch.sum(torch.pow(x-torch.mean(x),2), dim=1))/torch.numel(x)
    
    def __str__(self) -> str:
        return "RootMeanSquare"