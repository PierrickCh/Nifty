import torch


device= torch.device("cpu") # or "xpu" for XPU
def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(x)
    return a + b
opt_foo1 = torch.compile(foo)

print(opt_foo1(torch.randn(10, 10).to(device), torch.randn(10, 10).to(device)))
