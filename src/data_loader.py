import math
import numpy as np
from skimage.filters import gaussian
from torch.utils.data import Dataset

class CocoDataset(Dataset):
    def __init__(self,coco_train,opt):
        self.coco_train = coco_train
        self.num_of_keypoints = opt.num_of_keypoints
        self.anns = self.get_anns(self.coco_train)
        self.bbox_height = opt.coeff *28
        self.bbox_width = opt.coeff *18
        self.threshold = opt.threshold

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, item):
        ann_data = self.anns[item]

        input, label = self.get_data(ann_data, self.coco_train )

        return input, label

    def get_data(self, ann_data, coco):
        weights = np.zeros((self.bbox_height, self.bbox_width, 17))
        output = np.zeros((self.bbox_height, self.bbox_width, 17))

        bbox = ann_data['bbox']
        x = int(bbox[0])
        y = int(bbox[1])
        w = float(bbox[2])
        h = float(bbox[3])

        x_scale = float(self.bbox_width) / math.ceil(w)
        y_scale = float(self.bbox_height) / math.ceil(h)

        kpx = ann_data['keypoints'][0::3]
        kpy = ann_data['keypoints'][1::3]
        kpv = ann_data['keypoints'][2::3]


        for j in range(17):
            if kpv[j] > 0:
                x0 = int((kpx[j] - x) * x_scale)
                y0 = int((kpy[j] - y) * y_scale)

                if x0 >= self.bbox_width and y0 >= self.bbox_height:
                    output[self.bbox_height - 1, self.bbox_width - 1, j] = 1
                elif x0 >= self.bbox_width:
                    output[y0, self.bbox_width - 1, j] = 1
                elif y0 >= self.bbox_height:
                    try:
                        output[self.bbox_height - 1, x0, j] = 1
                    except:
                        output[self.bbox_height - 1, 0, j] = 1
                elif x0 < 0 and y0 < 0:
                    output[0, 0, j] = 1
                elif x0 < 0:
                    output[y0, 0, j] = 1
                elif y0 < 0:
                    output[0, x0, j] = 1
                else:
                    output[y0, x0, j] = 1

        img_id = ann_data['image_id']
        img_data = coco.loadImgs(img_id)[0]
        ann_data = coco.loadAnns(coco.getAnnIds(img_data['id']))

        for ann in ann_data:
            kpx = ann['keypoints'][0::3]
            kpy = ann['keypoints'][1::3]
            kpv = ann['keypoints'][2::3]

            for j in range(17):
                if kpv[j] > 0:
                    if (kpx[j] > bbox[0] - bbox[2] * self.threshold and kpx[j] < bbox[0] + bbox[2] * (1 + self.threshold)):
                        if (kpy[j] > bbox[1] - bbox[3] * self.threshold and kpy[j] < bbox[1] + bbox[3] * (1 + self.threshold)):
                            x0 = int((kpx[j] - x) * x_scale)
                            y0 = int((kpy[j] - y) * y_scale)

                            if x0 >= self.bbox_width and y0 >= self.bbox_height:
                                weights[self.bbox_height - 1, self.bbox_width - 1, j] = 1
                            elif x0 >= self.bbox_width:
                                weights[y0, self.bbox_width - 1, j] = 1
                            elif y0 >= self.bbox_height:
                                weights[self.bbox_height - 1, x0, j] = 1
                            elif x0 < 0 and y0 < 0:
                                weights[0, 0, j] = 1
                            elif x0 < 0:
                                weights[y0, 0, j] = 1
                            elif y0 < 0:
                                weights[0, x0, j] = 1
                            else:
                                weights[y0, x0, j] = 1

        for t in range(17):
            weights[:, :, t] = gaussian(weights[:, :, t])
        output = gaussian(output, sigma=2, mode='constant', multichannel=True)
        # weights = gaussian_multi_input_mp(weights)
        # output = gaussian_multi_output(output)
        return weights, output

    def get_anns(self, coco):
        #:param coco: COCO instance
        #:return: anns: List of annotations that contain person with at least 6 keypoints
        ann_ids = coco.getAnnIds()
        anns = []
        for i in ann_ids:
            ann = coco.loadAnns(i)[0]
            if ann['iscrowd'] == 0 and ann['num_keypoints'] > self.num_of_keypoints:
                anns.append(ann)  # ann
        sorted_list = sorted(anns, key=lambda k: k['num_keypoints'], reverse=True)
        return sorted_list
