import os, time
from tqdm import tqdm
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import ot
from PIL import Image
from torch import nn


if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    device = torch.device('cuda:0')
else :
    device = torch.device('cpu')



def Tensor_load(file_name):
    img = Image.open(file_name).convert('RGB')  # Ensure 3 channels (no alpha)
    img = torch.from_numpy(np.array(img)).float() / 255.0  # Normalize to [0, 1]
    img = img.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C, H, W)
    return img * 2 - 1  # Scale to [-1, 1]


def Tensor_display(img1, img2=None, title=None, pad_value=1.0):
    def to_np(img):
        if img.dim() == 4:
            img = img.squeeze(0)
        img = (img.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]
        return img.permute(1, 2, 0).cpu().numpy()

    def center_pad(img, target_h, target_w, pad_val):
        _, _, h, w = img.shape
        pad_h = target_h - h
        pad_w = target_w - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        # pad format: (left, right, top, bottom)
        padding = (pad_left, pad_right, pad_top, pad_bottom)
        return F.pad(img, padding, mode='constant', value=pad_val)

    images = [img1]
    if img2 is not None:
        images.append(img2)

    # Find max height and width
    heights = [img.shape[-2] for img in images]
    widths = [img.shape[-1] for img in images]
    max_h, max_w = max(heights), max(widths)

    # Center pad images to max size
    images = [center_pad(img, max_h, max_w, pad_value) for img in images]

    n = len(images)
    fig, axs = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axs = [axs]

    for ax, img in zip(axs, images):
        np_img = to_np(img)
        ax.imshow(np_img, interpolation='none')
        ax.axis('off')
        ax.set_xlim([0, max_w])
        ax.set_ylim([max_h, 0])

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()



def imsave(s,x):    
    out = (x.squeeze(0).permute(1, 2, 0).cpu().numpy()*.5+.5).clip(0, 1)
    plt.imsave(s, out)



def Patch_extraction(img, patchsize, stride) :
    P = torch.nn.Unfold(kernel_size=patchsize, dilation=1, padding=0, stride=stride)(img) # Tensor with dimension 1 x 3*Patchsize^2 x Heigh*Width/stride^2
    return P.to(torch.float32)

def Patch_Average(P_synth, patchsize, stride, W, H, D, spotsize=1/4) : 
    # Gaussian weight for patch center

    w=torch.exp(-torch.linspace(-patchsize//2,patchsize//2,steps=patchsize).pow(2)/2/(patchsize*spotsize)**2).to(device)
    w=w.view(-1,1)*w.view(1,-1)
    w=w.repeat(3,1,1).view(-1)
    w/=w.sum()
    w=w.unsqueeze(0).unsqueeze(-1)

    synth = nn.Fold((W,H), patchsize, dilation=1, padding=0, stride=stride)(P_synth*w)
    count = nn.Fold((W,H), patchsize, dilation=1, padding=0, stride=stride)(P_synth*0+w) # normalization to sum to 1


    count= (count*(count!=0)+1.*(count==0))
    synth = synth /count
    return synth



def make_times(n_timestep , schedule='linear', t0=0,linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3,p=.3): 
    '''
    different time discretizations, 'quad' has smaller timesteps near t=0
    '''
    if  schedule == "linear":
        times = torch.linspace(t0,1,n_timestep+1) 
        
    elif schedule == "quad":
        times=torch.linspace(t0**.5,1,n_timestep+1)**2

    elif schedule == "cosine":
        times = (
            torch.linspace(torch.arcsin(torch.tensor(t0)**.5),math.pi / 2,n_timestep+1) 
        )

        times = torch.sin(times).pow(2)
        times = times / times[-1]
    return times


def Patch_topk(P_exmpl, P_synth, N_subsampling, k=10,mem=None) :
    N = P_exmpl.size(2)
    
    ## random subsampling
    Ns = np.min([N_subsampling,N])
    I = torch.randperm(N)
    I = I[0:Ns]
    
    # Distance matrix between synthesis patches, and sampled exemplar partches
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
        # first iteration, output top k, and store their indices
        top=topk
        mem=torch.take(I,ind)
        dists=-J
    else:

        # subsequent iterations, fetch previous topk from indices in memory, and compute new topk among previous topk + new candidates  

        X_mem=P_exmpl[:,:,mem]
        X=X_mem[0].permute(2,0,1)
        Y=P_synth
        X2 = (X**2).sum(1)
        Y2 = (Y**2).sum(1)
        D_mem=(Y2+X2-2*(X*Y).sum(1)).T

        Dcat=torch.cat((D_mem,-J),dim=1) # distances to previous topk + new candidates
        indcat=torch.cat((mem,torch.take(I,ind)),dim=1)
        
        # the following lines remove duplicates (by setting D to infinity at the indices of duplicates), if indices in memory are found again in new candidates

        ###
        sorted_indcat,sorted_indcat_ind=torch.sort(indcat,dim=1)

        del_mask = torch.zeros_like(indcat, dtype=torch.bool)
        del_mask[:,1:] = (sorted_indcat[:,1:] == sorted_indcat[:,:-1])
        
        _,inv_sorted_indcat_ind=torch.sort(sorted_indcat_ind,dim=1)
        del_mask=torch.take_along_dim(del_mask,inv_sorted_indcat_ind,dim=1)
        Dcat[del_mask]=torch.inf
        ###

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


def Nifty(img,im2=None,rs=1.,T=100,k=10,patchsize=16,stride=1,size=(256,256),octaves=1,renoise=.5,warmup=0,show=True,memory=True,seed=None,noise=None,spotsize=1/4,blend=False,blend_alpha=0.5,save=True):
    if seed is not None:
        torch.manual_seed(seed)

    H,W=size
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

        # Resize annd extract reference patches for the image and the optional second image
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

        if s==0: # Initialize at coarsest scale
            if noise is None:
                synth=torch.randn(b,c,int(H*2**-(octaves-1)),int(W*2**-(octaves-1))).cuda()
            else:
                synth=noise.cuda()
            t0=0
        else: # Upsample from previous scale and renoise
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

        for it in range(T): # ODE steps
            t=times[it]
            P_synth = Patch_extraction(synth, patchsize, stride)
            ## NN SEARCH
            
            if not memory:
                mem=None
                mem2=None


            P_topk, D ,mem = Patch_topk(P_exmpl*t, P_synth, N_subsampling,k=k,mem=mem)
            P_topk=P_topk/t # renorm
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
        
        if save:
            imsave('./results/gt_s%d.png'%s,img_resized*sigma+mu)
            imsave('./results/synth_s%d.png'%s,synth*sigma+mu)
    if show: 
        Tensor_display(img_resized*sigma+mu,synth*sigma+mu)
        


    return synth*sigma+mu # denormalize to zero mean


