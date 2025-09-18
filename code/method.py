import os, time
from tqdm import tqdm
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import ot
from scipy.optimize import root_scalar
from skimage.transform import resize
from torch import nn


if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    device = torch.device('cuda:0')
else :
    device = torch.device('cpu')

def imsave(s,x):    
    out = (x.squeeze(0).permute(1, 2, 0).cpu().numpy()*.5+.5).clip(0, 1)
    plt.imsave('./results/%s.png'%s, out)
    
def Tensor_display(img1, img2=None):
    """
    Display one or two images (torch tensors) side by side using subplots, even if their sizes mismatch.
    img1, img2: torch tensors with shape (1, C, H, W) or (C, H, W)
    """
    def to_numpy(img):
        if img.dim() == 4:
            img_np = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * .5 + .5
        else:
            img_np = img.permute(1, 2, 0).detach().cpu().numpy() * .5 + .5
        return img_np.clip(0, 1)

    if img2 is None:
        img_np = to_numpy(img1)
        plt.figure(figsize=(10, 10))
        plt.imshow(img_np, interpolation="bicubic")
        plt.axis('off')
        plt.show()
    else:
        img1_np = to_numpy(img1)
        img2_np = to_numpy(img2)
        h1, w1 = img1_np.shape[:2]
        h2, w2 = img2_np.shape[:2]
        if h1 != h2:
            img2_np = resize(img2_np, (h1, int(w2 * h1 / h2)), preserve_range=True, anti_aliasing=True)
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        axes[0].imshow(img1_np, interpolation="bicubic")
        axes[0].axis('off')
        axes[1].imshow(img2_np, interpolation="bicubic")
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

def Tensor_load(file_name) : 
    img_np0 = plt.imread(file_name)
    if img_np0.max()>1 : 
        img_np0 = img_np0/img_np0.max()

    img = torch.tensor(img_np0, device=device, requires_grad = False).permute(2, 0, 1).unsqueeze(0)
    if img.size(1) == 4 :
        img = img[:,:3,:,:]
    return img

def Patch_extraction(img, patchsize, stride) :
    P = torch.nn.Unfold(kernel_size=patchsize, dilation=1, padding=0, stride=stride)(img) # Tensor with dimension 1 x 3*Patchsize^2 x Heigh*Width/stride^2
    return P

def Patch_Average(P_synth, patchsize, stride, W, H, D, spotsize=1/4) : 
    # r = 0.8 in Kwatra
    

    # Gaussian weight for patch center

    w=torch.exp(-torch.linspace(-patchsize//2,patchsize//2,steps=patchsize).pow(2)/2/(patchsize*spotsize)**2).to(device)
    w=w.view(-1,1)*w.view(1,-1)
    w=w.repeat(3,1,1).view(-1)
    w/=w.sum()
    w=w.unsqueeze(0).unsqueeze(-1)

    synth = nn.Fold((W,H), patchsize, dilation=1, padding=0, stride=stride)(P_synth*w)
    count = nn.Fold((W,H), patchsize, dilation=1, padding=0, stride=stride)(P_synth*0+w)


    count= (count*(count!=0)+1.*(count==0))
    synth = synth /count
    return synth



def make_times(
    n_timestep , schedule='linear', t0=0,linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3,p=.3): # different time discretizations
    
    if schedule == "quad":
        times=torch.linspace(t0**.5,1,n_timestep+1)**2

    elif schedule == "linear":
        times = torch.linspace(t0,1,n_timestep+1) 

    elif schedule == "cosine":
        times = (
            torch.linspace(torch.arcsin(torch.tensor(t0)**.5),math.pi / 2,n_timestep+1) 
        )

        times = torch.sin(times).pow(2)
        times = times / times[-1]
    elif schedule == "poly":
        p=.2
        f=lambda x: 4*(1-p)*x**3 + 6*(p-1)*x**2 + (3-2*p)*x
        func = lambda x: f(x) - t0
        sol = root_scalar(func, bracket=[0,2], method='brentq')
        inv_t0 = sol.root 
        times = torch.linspace(inv_t0,1,n_timestep+1) 
        times=f(times)

    return times


def Patch_topk(P_exmpl, P_synth, N_subsampling, k=10,mem=None,proj=None,I=None) :
    N = P_exmpl.size(2)
    
    ## random subsampling
    Ns = np.min([N_subsampling,N])
    if I is None:
        I = torch.randperm(N)
        I = I[0:Ns]
    
    X = P_exmpl[:,:,I] 
    X = X.squeeze(0) # d x Ns
    X2 = (X**2).sum(0).unsqueeze(0) # 1 x Ns
    Y = P_synth.squeeze(0) # d x N
    Y2 = (Y**2).sum(0).unsqueeze(0) # squared norm : 1 x N
    D = Y2.transpose(1,0) - 2 * torch.matmul(Y.transpose(1,0),X) + X2 #N Ns



    J,ind = torch.topk(-D,k=k,dim=1)
    topk = X[:,ind].unsqueeze(0) 
    I=I.cuda()
    if mem is None:
        top=topk
        mem=torch.take(I,ind)
        dists=-J
    else:

        X_mem=P_exmpl[:,:,mem]
        X=X_mem[0].permute(2,0,1)
        Y=P_synth
        X2 = (X**2).sum(1)
        Y2 = (Y**2).sum(1)
        D_mem=(Y2+X2-2*(X*Y).sum(1)).T

        Dcat=torch.cat((D_mem,-J),dim=1)
        indcat=torch.cat((mem,torch.take(I,ind)),dim=1)
        
        sorted_indcat,sorted_indcat_ind=torch.sort(indcat,dim=1)

        del_mask = torch.zeros_like(indcat, dtype=torch.bool)
        del_mask[:,1:] = (sorted_indcat[:,1:] == sorted_indcat[:,:-1])
        
        _,inv_sorted_indcat_ind=torch.sort(sorted_indcat_ind,dim=1)
        del_mask=torch.take_along_dim(del_mask,inv_sorted_indcat_ind,dim=1)
        Dcat[del_mask]=torch.inf


        Dcat,dists_ind=torch.sort(Dcat,dim=-1)
        Dcat=Dcat[:,:k]
        new_mem=torch.take_along_dim(indcat,dists_ind[:,:k],dim=1) # doesn t avoid ducplicates, gives double mass to duplicates, allows stacking n over time :/
        topk=torch.take_along_dim(torch.cat((X_mem,topk),dim=-1),dists_ind[:,:k].unsqueeze(0).unsqueeze(0),dim=-1)

        top=topk
        dists=Dcat
        mem=new_mem

    # top k matches, their distances will be used to compute flow weights
    # mem : indices of the patches in the exemplar stored for next iteration
        
    return top, dists, mem


def Nifty(img,im2=None,rs=1.,T=100,k=10,patchsize=16,stride=1,size=(256,256),octaves=1,renoise=.5,warmup=0,show=True,memory=True,seed=None,noise=None,spotsize=1/4,blend=False,blend_alpha=0.5):
    if seed is not None:
        torch.manual_seed(seed)

    H,W=size
    h,w=0,0
    b,c,_,_=img.shape

    if im2 is not None and not blend:
        imcat=torch.cat((img.view(1,3,-1),im2.view(1,3,-1)),dim=-1)
        mu,sigma=imcat.mean(),imcat.std()
        img=(img-mu)/sigma 
        im2=(im2-mu)/sigma 
    elif im2 is not None:
        imcat=torch.cat((img.view(1,3,-1),im2.view(1,3,-1)),dim=-1)
        mu,sigma=imcat.mean(),imcat.std()
        img=(img-mu)/sigma 
        im2=(im2-mu)/sigma 
    else:
        mu,sigma=img.mean(),img.std()
        img=(img-mu)/sigma 

    for s in range(octaves):
        mem=None
        mem2=None
        if s==(octaves-1):
            img_resized=img
            if im2 is not None:
                im2_resized=im2
        else:
            img_resized=F.interpolate(img,size=(int(img.shape[-2]*2**-(octaves-1-s)),int(img.shape[-1]*2**-(octaves-1-s))),mode='bicubic')
            if im2 is not None:
                im2_resized=F.interpolate(im2,size=(int(im2.shape[-2]*2**-(octaves-1-s)),int(im2.shape[-1]*2**-(octaves-1-s))),mode='bicubic')
        if im2 is not None and not blend:
            P_exmpl = torch.cat((Patch_extraction(img_resized,patchsize=patchsize,stride=stride), Patch_extraction(im2_resized,patchsize=patchsize,stride=stride)),dim=-1) 
        elif im2 is not None: # blend
            P_exmpl = Patch_extraction(img_resized,patchsize=patchsize,stride=stride)
            P_exmpl2 = Patch_extraction(im2_resized,patchsize=patchsize,stride=stride)
        else:
            P_exmpl = Patch_extraction(img_resized,patchsize=patchsize,stride=stride) #

        N_subsampling=int(rs*P_exmpl.shape[-1])

        if s==0:
            if noise is None:
                synth=torch.randn(b,c,int(H*2**-(octaves-1)),int(W*2**-(octaves-1))).cuda()
            else:
                synth=noise.cuda()
            t0=0
        else:
            synth=F.interpolate(synth,size=(int(H*2**-(octaves-1-s)),int(W*2**-(octaves-1-s))),mode='bicubic')
            t0=renoise
            synth=synth*t0+torch.randn(synth.shape).cuda()*(1-t0)
        
        if t0!=0:
            times=make_times(T,t0=t0,schedule='linear')
        else:
            times=make_times(T+1,t0=0,schedule='linear')[1:]
            P_synth = Patch_extraction(synth, patchsize, stride)
            mean_ref = P_exmpl.mean(dim=-1,keepdim=True)
            P_flow=mean_ref-P_synth # flow first step, avoid 0 division, go towards mean patch
            flow = Patch_Average(P_flow, patchsize, stride,  synth.shape[-2], synth.shape[-1], 0,spotsize=spotsize)
            synth+=flow*times[0]
        

        if memory==True and s!=0:
            #optional warmup to accumulate good matches in memory before starting synthesis
            P_synth = Patch_extraction(synth, patchsize, stride)

            for _ in range(warmup):
                P_topk, D ,mem = Patch_topk(P_exmpl*t0, P_synth, N_subsampling,k=k,mem=mem)

            if show and warmup>0:
                plt.plot(l)
                plt.show()
      

  
        for it in range(T):
            t=times[it]

            P_synth = Patch_extraction(synth, patchsize, stride)
            
            ## NN SEARCH
            
            if not memory:
                mem=None
                mem2=None


            P_topk, D ,mem = Patch_topk(P_exmpl*t, P_synth, N_subsampling,k=k,mem=mem)
            P_topk=P_topk/t# renorm


            weight=nn.Softmax(dim=1)(-D/2/(1-t)**2) # flow weights
            P_flow=((P_topk-P_synth.unsqueeze(-1))*weight.unsqueeze(0).unsqueeze(0)).sum(-1)/(1-t) # \hat{\omega}} in the paper

            if im2 is not None and blend:
                P_topk2, D ,mem2 = Patch_topk(P_exmpl2*t, P_synth, N_subsampling,k=k,mem=mem2)
                P_topk2=P_topk2/t
                weight2=nn.Softmax(dim=1)(-D/2/(1-t)**2)
                P_flow2=((P_topk2-P_synth.unsqueeze(-1))*weight2.unsqueeze(0).unsqueeze(0)).sum(-1)/(1-t)
                P_flow=blend_alpha*P_flow+(1-blend_alpha)*P_flow2


            P_synth += P_flow*(times[it+1]-t) # ODE steps and aggregation of flows
            synth = Patch_Average(P_synth, patchsize, stride,  synth.shape[-2], synth.shape[-1], D[:,0],spotsize=spotsize) 
        if show: 
            Tensor_display(synth*sigma+mu,img_resized*sigma+mu)
        imsave('gt_s%d'%s,img_resized*sigma+mu)
        imsave('synth_s%d'%s,synth*sigma+mu)
        


    return synth*sigma+mu # denormalize to zero mean


