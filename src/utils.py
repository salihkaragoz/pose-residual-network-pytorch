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


def save_ckpt(state, ckpt_path, is_best=True):
    if is_best:
        file_path = os.path.join(ckpt_path, 'ckpt_best.pth.tar')
        torch.save(state, file_path)
    else:
        file_path = os.path.join(ckpt_path, 'ckpt_last.pth.tar')
        torch.save(state, file_path)

