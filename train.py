import os
from tqdm import tqdm
from progress.bar import Bar
from pycocotools.coco import COCO

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from opt import Options
from src.model import PRN
from src.eval import Evaluation
from src.utils import save_ckpt
from src.utils import save_options
from src.data_loader import CocoDataset


def main(optin):
    if not os.path.exists('checkpoint/'+optin.exp):
        os.makedirs('checkpoint/'+optin.exp)

    model = PRN(optin.node_count,optin.coeff).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=optin.lr)
    criterion = torch.nn.BCELoss().cuda()

    print model
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    save_options(optin, os.path.join('checkpoint/' + optin.exp), model.__str__(), criterion.__str__(), optimizer.__str__())

    print ('---------Loading Coco Training Set--------')
    coco_train = COCO(os.path.join('data/annotations/person_keypoints_train2017.json'))
    trainloader = DataLoader(dataset=CocoDataset(coco_train,optin),batch_size=optin.batch_size, num_workers=optin.num_workers, shuffle=True)

    bar = Bar('-->', fill='>', max=len(trainloader))

    err_best = 1000
    cudnn.benchmark = True
    for epoch in range(optin.number_of_epoch):
        print ('-------------Training Epoch {}-------------'.format(epoch))
        print 'Total Step:', len(trainloader), '| Total Epoch:', optin.number_of_epoch
        for idx, (input, label) in tqdm(enumerate(trainloader)):

            input = input.cuda().float()
            label = label.cuda().float()

            outputs = model(input)

            optimizer.zero_grad()
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            if idx % 200 == 0:
                bar.suffix = 'Epoch: {epoch} Total: {ttl} | ETA: {eta:} | loss:{loss}' \
                .format(ttl=bar.elapsed_td, eta=bar.eta_td, loss=loss.data, epoch=epoch)
                bar.next()


        Evaluation(model, optin)

        err_test = loss.data
        is_best = err_test < err_best
        if is_best:
            save_ckpt({'epoch': epoch + 1,
                           'err': err_best,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          ckpt_path='checkpoint/'+optin.exp,
                          is_best=True)
        else:
            save_ckpt({'epoch': epoch + 1,
                           'err': err_best,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          ckpt_path='checkpoint/'+optin.exp,
                          is_best=False)

        model.train()

if __name__ == "__main__":
    option = Options().parse()
    main(option)
