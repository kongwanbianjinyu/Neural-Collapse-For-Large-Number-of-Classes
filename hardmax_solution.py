import numpy as np
import torch
import matplotlib.pyplot as plt

def hardmax(W,H):
  d_dist = W @ H.T
  wh = torch.diag(d_dist)
  matrix = d_dist - wh.unsqueeze(1).repeat((1,W.shape[0]))
  for i in range(W.shape[0]):
      matrix[i, i] = -np.inf
  max = torch.max(matrix)
  return max

def minimize(W, H, alpha=0.1, tol=1e-6, max_iter=100000):
    """
    Use gradient descent to minimize the objective function.
    """
    lr_sched = np.linspace(0, alpha, num=max_iter)
    lr_sched = lr_sched[::-1]
    W = torch.autograd.Variable(W, requires_grad=True)
    H = torch.autograd.Variable(H, requires_grad=True)
    for i in range(max_iter):
        f = hardmax(W, H)
        # f = torch.nn.functional.cross_entropy(W@H.T*1, torch.arange(0, W.shape[0]).type(torch.LongTensor).to(W.device))
        f.backward()
        if torch.norm(W.grad) < tol and torch.norm(H.grad):
            break
        with torch.no_grad():
            W -= lr_sched[i] * W.grad
            W /= torch.norm(W, dim=1, keepdim=True)
            W.grad.zero_()
            H -= lr_sched[i] * H.grad
            H /= torch.norm(H, dim=1, keepdim=True)
            H.grad.zero_()
        if i%5000 == 0:
          print("iteration " + str(i).zfill(7) +" lr: %.3f"%lr_sched[i]+" f_value: %.8f" %f.item() + " max difference: %.5f"%torch.max(torch.norm(W-H, dim=1)).item())
    return f, W, H


if __name__ == "__main__":
    d_K_pair = [(3,12)]#[(21,162),(22,100),(8,240)]#[(21,162)]#[(22,100)] #[(22,100),(21,162)]
    cos_list = []
    lr = 0.1
    device = "cuda:0"
    for (d,K) in d_K_pair:
        print(f"d: {d}, K: {K}")
        W = torch.randn((K, d)).to(device)
        #W_np = np.load("./WWT_matrix/d22_K100.npy")
        #W = torch.tensor(W_np).to(device)
        W /= torch.norm(W, dim=1, keepdim=True)
        H = W

        init_f= hardmax(W, H)
        print('init_f: ', init_f)
        minimizer, W, H = minimize(W, H, alpha=lr)


        WWT = (W @ W.T).detach().cpu().numpy()
        with open(f'./WWT_matrix/d{d}_K{K}.npy', 'wb') as f:
            np.save(f, WWT)

        for i in range(WWT.shape[0]):
            WWT[i,i] = -np.inf
        print("max cosine value:", np.max(WWT))
        cos_list.append(np.max(WWT))

    print(cos_list)