import os
import json
import torch

def save_options(opt, path,model,criterion, optimizer):
    file_path = os.path.join(path, 'opt.json')
    model_struc = model.__str__()
    model_struc = {'Model': model_struc, 'Loss Function': criterion, 'Optimizer': optimizer}

    with open(file_path, 'w') as f:
        f.write(json.dumps(vars(opt), sort_keys=True, indent=4))
        f.write(json.dumps(model_struc, sort_keys=True, indent=4))


def save_model(state, checkpoint, filename='checkpoint.pth.tar'):
    filename = 'epoch'+str(state['epoch']) + filename
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def adjust_lr(optimizer, epoch, gamma):
    schedule = list(range(3,32,2))
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma
    return optimizer.state_dict()['param_groups'][0]['lr']
