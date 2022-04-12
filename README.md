## Brax Halfcheetah Exploding Gradients

The problem can be reproduced by using the official implementation analytical policy gradient [apg](https://github.com/google/brax/blob/main/brax/training/apg.py) with official reward function.
The only thing I changed is to print out the gradient norm before clipping.

#### To reproduce
```
python apg.py
```

#### Environment
```
python                    3.8

brax                      0.0.12
jax                       0.3.5                    
jaxlib                    0.3.5+cuda11.cudnn82    
```

#### nvidia-smi
```
NVIDIA-SMI 510.54       Driver Version: 510.54       CUDA Version: 11.6`
```

#### Gradient norm from Halfcheetah
```
grad_raw [inf]
grad_raw [inf]
grad_raw [inf]
grad_raw [3.7764926e+18]
grad_raw [inf]
grad_raw [inf]

```

#### Gradient norm from ant
```
grad_raw [1.7340995]
grad_raw [2.4045153]
grad_raw [2.8107145]
grad_raw [1.8724597]
grad_raw [3.0794723]
grad_raw [2.4992204]
```