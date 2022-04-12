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
grad_raw [1.44705e+13]
grad_raw [7.1334626e+16]
grad_raw [inf]
grad_raw [5.04371e+13]
grad_raw [2.6387382e+10]
grad_raw [1.6068858e+16]
```

#### Gradient norm from ant
```
grad_raw [1.4597763]
grad_raw [8.197796]
grad_raw [3.641092]
grad_raw [1.9890647]
grad_raw [4.062545]
```