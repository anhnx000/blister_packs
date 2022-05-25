# import tensorflow as tf 

import wandb 
wandb.init(project="xanh_geo")
train_acc = 0 
train_loss  = 1
wandb.log({'accuracy': train_acc, 'loss': train_loss})



