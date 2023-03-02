from .trainer import train


class ClassicTrainer(object):
    def __init__(self):pass
    
    def train(self, args, listner, train_env, val_envs, aug_env=None, rank=-1):
        ''' Train the model. '''
        return train(args, listner, train_env, val_envs, aug_env=aug_env, rank=rank)
