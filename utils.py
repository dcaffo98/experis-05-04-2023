import torch
import os

def get_device():
    if int(os.environ.get('FORCE_CPU', 0)):
        device = 'cpu'
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device {device}")
    return device
    
COLORS = (
    'green',
    'red',
    'blue',
    'white',
    'black',
    'gray',
    'orange',
)