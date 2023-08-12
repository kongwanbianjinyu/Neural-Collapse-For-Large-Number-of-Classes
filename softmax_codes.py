
import numpy as np
import torch
from GNC2 import *


# Tammes
def Tammes_loss_func(W):
    Cov_W = W @ W.T
    for i in range(Cov_W.shape[0]):
        Cov_W[i, i] = -np.inf
    loss = torch.max(Cov_W)
    return loss



def minimize(W, tau=1, alpha=0.01, tol=1e-6, max_iter=10):
    """
    Use gradient descent to minimize the objective function.
    """
    d = W.shape[1]
    lr_sched = np.linspace(0, alpha, num=max_iter)
    lr_sched = lr_sched[::-1]
    W = torch.autograd.Variable(W, requires_grad=True)

    for i in range(max_iter):
        # f = CE_loss_func(W, tau = tau)
        f = Tammes_loss_func(W)
        f.backward()
        if torch.norm(W.grad) < tol:
            break
        #if i % 500 == 0:
        print("iteration " + str(i).zfill(7) + " lr: %.3f" % lr_sched[
                i] + " f_value: %.8f" % f.item() + " grad_norm: %.5f" % torch.norm(W.grad).item())

        with torch.no_grad():
            W -= lr_sched[i] * W.grad
            W /= torch.norm(W, dim=1, keepdim=True)
            W.grad.zero_()
    return f, W

if __name__ == "__main__":
    d = 512
    K = 10000
    lr = 0.01
    device = "cuda:2"
    print(f"{device}-d-{d}, K-{K}:")
    W = torch.randn((K, d)).to(device)
    W /= torch.norm(W, dim=1, keepdim=True)

    minimizer, W = minimize(W, alpha=lr)

    W = W.detach().cpu().numpy()

    with open(f'W-d{d}-K{K}-iter10.npy', 'wb') as f:
        np.save(f, W)

    # with open(f'W-d{d}-K{K}.npy', 'rb') as f:
    #     W = np.load(f)

    NC2_W = compute_NC2_matrix_form(device, W)
    print(f"d-{d}, K-{K} Min distance to convex hull is: {abs(NC2_W.item())}")
