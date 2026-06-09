## Variables

rs: Ratio of the reference patches to sample at each step.

T: Number of (linear) discretization steps between 0 and 1 to solve the flow ODE.

k: Number of top closest patch used to approximate the velocity field.

octaves: Number of diadic scales used for the synthesis.

renoise: 'time' $t_r$ used renoise the smooth upsampled image at each resolution: $x_{start}=t_r*x_{upsampled}+(1-t_r)z$. In particular, $t_r=1$ means  additional noise.