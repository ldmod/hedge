
import torch
from torch.nn import Parameter
import torch.autograd as autograd
import torch.nn as nn
from alpha.tools import *

class _f_matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, A):
        out = torch.matmul(X, A)
        ctx.save_for_backward(X, A)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        import pdb; pdb.set_trace()
        X, A = ctx.saved_tensors
        # 矩阵求导部分
        grad_X = torch.matmul(out_grad, A.T)
        grad_A = torch.matmul(X.transpose(-2,-1), out_grad)
        return grad_X, grad_A

f_mat_mul = _f_matmul.apply

class Zhihu_MatMul(torch.nn.Module):
    def __init__(self, A: torch.Tensor):
        super(Zhihu_MatMul, self).__init__()
        self.A = Parameter(data=A, requires_grad=True)

    def forward(self, X: torch.Tensor):
        out = f_mat_mul(X, self.A)
        return out

class Torch_Matmul(torch.nn.Module):
    def __init__(self, A: torch.Tensor):
        super(Torch_Matmul, self).__init__()
        self.A = Parameter(data=A, requires_grad=True)

    def forward(self, X: torch.Tensor):
        out = torch.matmul(X, self.A)
        return out
    
def _is_equal(x: torch.Tensor, y: torch.Tensor, tol=1e-2):
    diff = abs(x - y)
    x_max = torch.max(x)
    y_max = torch.max(y)
    err = torch.max(diff) / torch.max(x_max, y_max)
    return err <= tol

def testmm():
    # 初始化A，X以及输出的Y
    X = torch.randn((2, 8, 10))
    A = torch.randn((10, 5))
    torch_module = Torch_Matmul(A)
    zhihu_module = Zhihu_MatMul(A)
    # 前向计算（都是直接用的torch.matmul(X, A))，所以肯定相等
    torch_out = torch_module(X)
    zhihu_out = zhihu_module(X)
    if _is_equal(torch_out, zhihu_out):
        print("Torch_Matmul and Zhihu_MatMul match (Forward)")
    else:
        print("Torch_Matmul and Zhihu_MatMul differ (Forward)")
    # 计算Loss
    torch_loss = torch.sum(torch_out)
    zhihu_loss = torch.sum(zhihu_out)
    # 后向计算（Zhihu_MatMul是我们手动写的反向过程）
    torch_A_grad = autograd.grad(torch_loss, torch_module.A, retain_graph=True)[0]
    zhihu_A_grad = autograd.grad(zhihu_loss, zhihu_module.A, retain_graph=True)[0]
    if _is_equal(torch_A_grad, zhihu_A_grad):
        print("Torch_Matmul and Zhihu_MatMul match (Backward)")
    else:
        print("Torch_Matmul and Zhihu_MatMul differ (Backward)")
        
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
     
def testcss():
    # 初始化A，X以及输出的Y
    setup_seed(0)
    X = torch.randn((2, 8, 10))
    A = torch.randn((10, 5))
    setup_seed(0)
    torch_module = CssModel(10, 5)
    setup_seed(0)
    zhihu_module = OptCssModel(10,5)
    # 前向计算（都是直接用的torch.matmul(X, A))，所以肯定相等
    torch_out = torch_module(X)
    zhihu_out = zhihu_module(X)
    if _is_equal(torch_out, zhihu_out):
        print("Torch_Matmul and Zhihu_MatMul match (Forward)")
    else:
        print("Torch_Matmul and Zhihu_MatMul differ (Forward)")
    # 计算Loss
    torch_loss = torch.sum(torch_out)
    zhihu_loss = torch.sum(zhihu_out)
    # 后向计算（Zhihu_MatMul是我们手动写的反向过程）
    torch_A_grad = autograd.grad(torch_loss, torch_module.hidden1[0].weight, retain_graph=True)[0]
    zhihu_A_grad = autograd.grad(zhihu_loss, zhihu_module.hidden1.weight, retain_graph=True)[0]
    if _is_equal(torch_A_grad, zhihu_A_grad):
        print("Torch_Matmul and Zhihu_MatMul match (Backward)")
    else:
        print("Torch_Matmul and Zhihu_MatMul differ (Backward)")
if __name__ == '__main__':
    testcss()