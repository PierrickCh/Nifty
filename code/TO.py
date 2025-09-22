# code from JULIEN RABIN
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time


import os


##hyperparams:
N_subsampling = int(10000000) # random sampling of patch in the input image to reduce the memory footprint (put to infinity to use all patches)
scale_size = [1/4, 1/2, 1] 
patch_size = [[8],[16, 8],[32, 16, 8]] 
PLOT=False
niter=11
imsize=512

if True and torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    #torch.cuda.set_device(1)
    #torch.cuda.device(0)
    device = torch.device('cuda:0')
    # torch.cuda.current_device()
else :
    pass
    dtype = torch.FloatTensor
    device = torch.device('cpu')



def Tensor_display(img_torch) : # display images coded as torch tensor
    img_np = img_torch.squeeze(0).permute(1, 2, 0).cpu().numpy() #is an array, shaped as an image for plt with permute

    plt.figure()
    plt.imshow(img_np, interpolation="bicubic")
    plt.show()
    
def Tensor_load(file_name) : # load an image as a torch tensor : BATCHSIZE=1 x COLOR=3 x Width x Height
    img_np0 = plt.imread(file_name)
    if img_np0.max()>1 : # normalization pour corriger un bug entre PNG (dans [0,1]) et JPG (dans [0,255])
        img_np0 = img_np0/img_np0.max()

    img_torch = torch.tensor(img_np0, dtype = torch.float, device=device, requires_grad = False).permute(2, 0, 1).unsqueeze(0)
    return img_torch

## patch nearest neighbor search
def Patch_NN_search(P_exmpl, P_synth, N_subsampling) :
    N = P_exmpl.size(2)
        
    ## precomputation for NN search
    Ns = np.min([N_subsampling,N])
    
    I = torch.randperm(N)#.to(device)
    I = I[0:Ns]
    X = P_exmpl[:,:,I] #.to(device) # 1 x d x Ns
    X = X.squeeze(0) # d x Ns
    X2 = (X**2).sum(0).unsqueeze(0) # 1 x Ns

    ## NN SEARCH
    Y = P_synth.squeeze(0) # d x N
    Y2 = (Y**2).sum(0).unsqueeze(0) # squared norm : 1 x N
    D = Y2.transpose(1,0) - 2 * torch.matmul(Y.transpose(1,0),X) + X2
    
    
    J = torch.argmin(D,1)
    #P_synth = P_exmpl[:,:,I[J]] # patch matching
    P_synth = X[:,J].unsqueeze(0) # same
    
    D = torch.min(D,1)[0] # squared distance
    return P_synth, D

## extract patch from an image
def Patch_extraction(img_torch, patchsize, stride) :
    P = torch.nn.Unfold(kernel_size=patchsize, dilation=1, padding=0, stride=stride)(img_torch) # Tensor with dimension 1 x 3*Patchsize^2 x Heigh*Width/stride^2
    return P

## patch aggregation into an image by weighted averaging
def Patch_Average(P_synth, patchsize, stride, W, H, r, D) : 
    # r = 0.8 in Kwatra
    
    if r==2 : # simple average with L2^2 distance
        synth = nn.Fold((W,H), patchsize, dilation=1, padding=0, stride=stride)(P_synth)
        count = nn.Fold((W,H), patchsize, dilation=1, padding=0, stride=stride)(P_synth*0+1)

    else : # weighted average using least square reweighting
        count = torch.pow(torch.max(D,torch.zeros_like(D)+1e-8), (r-2.)/2.)
        count = count.view(1,1,-1).repeat(1,P_synth.size(1),1)

        D = torch.pow(torch.max(D,torch.zeros_like(D)), r/2)

        synth = nn.Fold((W,H), patchsize, dilation=1, padding=0, stride=stride)(P_synth * count)
        count = nn.Fold((W,H), patchsize, dilation=1, padding=0, stride=stride)(count)

    synth = synth / count
    return synth


## synthesis initialization
def initialization(img_torch, block_size = 1) :
    if block_size==1 : # random init by permuting color pixel
        synth = torch.clone(img_torch)

        synth = synth.view(1,3,-1)
        tmp   = img_torch.view(1,3,-1)
        I = torch.randperm(tmp.size(2))
        synth[0,0,:] = tmp[0,0,I]
        synth[0,1,:] = tmp[0,1,I]
        synth[0,2,:] = tmp[0,2,I]

        synth = synth.view(img_torch.size())

    else : # random permutation of patchs (à la manière d'un taquin)
        size_init = block_size
        stride_init = size_init # size_init//2 # patch sans superposition si egal a size_init

        # 
        P_synth = torch.nn.Unfold(kernel_size=size_init, dilation=1, padding=0, stride=stride_init)(img_torch)
        P_synth = P_synth[:,:,torch.randperm(P_synth.size(2))]
        synth = nn.Fold((img_torch.size(2), img_torch.size(3)), size_init, dilation=1, padding=0, stride=stride_init)(P_synth)

        count = nn.Fold((img_torch.size(2), img_torch.size(3)), size_init, dilation=1, padding=0, stride=stride_init)(P_synth*0+1)
        synth = synth / count
        
    return synth

def initialization_scale(img_torch, block_size = 1, scale_factor = 2) : # cas où la sortie est plus grande que la texture d'entrée
    if int(scale_factor)==1 :
        return initialization(img_torch, block_size = block_size)
    
    scale_factor = int(scale_factor)
    if block_size==1 : # random init by permuting color pixel
        synth = torch.clone(img_torch)
        synth = synth.repeat(1, 1, scale_factor, scale_factor)
        
        synth = synth.view(1,3,-1)
        tmp   = synth.view(1,3,-1)
        I = torch.randperm(tmp.size(2))
        synth[0,0,:] = tmp[0,0,I]
        synth[0,1,:] = tmp[0,1,I]
        synth[0,2,:] = tmp[0,2,I]

        synth = synth.view(img_torch.size())

    else : # random permutation of patchs (à la manière d'un taquin)
        
        synth = torch.clone(img_torch)
        synth = synth.repeat(1, 1, scale_factor, scale_factor)
        
        size_init = block_size
        stride_init = size_init # size_init//2 # patch sans superposition si egal a size_init

        # 
        P_synth = torch.nn.Unfold(kernel_size=size_init, dilation=1, padding=0, stride=stride_init)(synth)
        P_synth = P_synth[:,:,torch.randperm(P_synth.size(2))]
        synth = nn.Fold((synth.size(2), synth.size(3)), size_init, dilation=1, padding=0, stride=stride_init)(P_synth)

        count = nn.Fold((synth.size(2), synth.size(3)), size_init, dilation=1, padding=0, stride=stride_init)(P_synth*0+1)
        synth = synth / count
        
    return synth

def TextureOptimization(img_torch0,N_subsampling,output_size=256,init='') :
    
    loss = np.array([]);
    Iter = 0;

    ## Algorithm
    for it_scale,scale in enumerate(scale_size) :
        #print('@ scale = ', it_scale, ' with resolution = ', scale)

        ## downsample input image 
        if scale<1 :
            img_torch = F.interpolate(img_torch0, size=int(imsize * scale), scale_factor=None, mode='bicubic', align_corners=False).clamp(0,1)
        else :
            img_torch = img_torch0
        if PLOT: Tensor_display(img_torch)


        ## INIT
        if it_scale==0 : # random init
            #synth = initialization(img_torch, 8) # if block_size = 1 : color permutation, if block_size = 4 permutation of small patch
            
            if init=='noise:':
                synth=torch.randn(1,3,int(output_size * scale),int(output_size * scale),device=device)*img_torch.std()+img_torch.mean()
            else:
                synth = initialization_scale(img_torch, 8, scale_factor=int(output_size / imsize))
        else : # init from previous scale 
            synth = F.interpolate(synth, size=int(output_size * scale), scale_factor=None, mode='bicubic', align_corners=False).clamp(0,1)



        for it_patch,patchsize in enumerate(patch_size[it_scale]) :    
            #print('... @ patch resolution = ', it_patch, ' with patch size = ', patchsize)    
            #patchsize = 8 # patch size P
            stride = patchsize//4 # patch stride
            #patchdim = size**2*3 # patch dim

            ## Patch extraction
            P_exmpl = Patch_extraction(img_torch, patchsize, stride) # 1 x 3P^2 x HW/stride^2

            # Mono-scale ALGORITHM
            for it in range(niter):
                Iter = Iter+1;
                #print('iter = ', Iter)

                ## Patch extraction
                P_synth = Patch_extraction(synth, patchsize, stride)

                ## NN SEARCH
                P_synth, D = Patch_NN_search(P_exmpl, P_synth, N_subsampling) 

                ## patch Aggregation
                r = 0.8; W = synth.size(2); H = synth.size(3)
                synth = Patch_Average(P_synth, patchsize, stride, W, H, r, D)

                ## print loss
                loss = np.append(loss, D.mean(0).cpu().numpy()) # for display after optimization
                #print('loss [', Iter, '] = ',loss[-1])

                ## Plot image
                if PLOT and it%10==0 : Tensor_display(synth)
    return synth, loss