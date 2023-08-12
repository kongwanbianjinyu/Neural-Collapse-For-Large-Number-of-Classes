import pytorch_lightning as pl
from args import *
import torch
import torch.nn as nn
import os
from ModelModule import *
from tqdm import tqdm

def loss_func(W, H):
    logits = W @ H.T
    WH = torch.diag(logits).unsqueeze(0)
    logits -= WH
    logits -= 100 * torch.eye(W.shape[0]).to(W.device)
    max_logits = torch.max(logits, dim=0, keepdim=True)[0]
    return max_logits

def minimize(W, H, lr=0.01, max_iter=10000):

    """
    Use gradient descent to minimize the objective function.
    """
    # lr_sched = np.linspace(0, lr, num=max_iter)
    # lr_sched = lr_sched[::-1]
    lr_step_sched = max_iter // 5
    H = torch.autograd.Variable(H.to(W.device), requires_grad=True)
    for i in tqdm(range(max_iter)):
        if (i+1) % lr_step_sched == 0:
            lr *= 0.1
        f = loss_func(W, H)
        f_sum = f.sum()
        f_sum.backward()
        with torch.no_grad():
            H -= lr * H.grad
            H /= torch.norm(H, dim=1, keepdim=True)
            H.grad.zero_()
    return f, H

def compute_NC2_matrix_form(device,W):
    K, d = W.shape
    W = torch.tensor(W).to(device)
    W = W.detach()
    H = torch.randn([K,d], device=device)
    H /= torch.norm(H, dim=1, keepdim=True)
    fs, H= minimize(W, H)

    distance = torch.min(-fs)
    return distance




def main(args):
    # args
    pl.seed_everything(0)
    print(args)

    nc2_list = []

    for i in range(20):
    #i = 19
        print(f"Loading checkpoint from {args.ckpt_path}: epoch={i*10+9}.ckpt")
        ckpt_path = os.path.join(args.ckpt_path, f"epoch={i*10+9}.ckpt")
        model = ModelModule.load_from_checkpoint(ckpt_path)

        weights = model.linear.weight.to(args.device)

        W = nn.functional.normalize(weights, dim=1, p=2)
        W = W.detach().cpu().numpy()

        NC2_W = compute_NC2_matrix_form(args.device, W)

        nc2_list.append(NC2_W.item())
        print(f"epoch: {i*10+9}, Min distance to convex hull is: {abs(NC2_W.item())}")


    print("NC2 list:", nc2_list)



if __name__ == '__main__':
    args = get_eval_arguments()
    main(args)