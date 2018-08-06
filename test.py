import argparse
import torch
from opt import Options
from src.eval import Evaluation
from src.model import PRN



if __name__ == "__main__":
    option = Options().parse()
    
    model = PRN(option.node_count, option.coeff).cuda()
    checkpoint = torch.load(option.test_cp)
    model.load_state_dict(checkpoint['state_dict'])

    Evaluation(model, option)

 
