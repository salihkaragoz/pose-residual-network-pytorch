**WARNING:** _This repository is under active construction._

# Pose Residual Network

This is the code for the paper:

Muhammed Kocabas, Salih Karagoz, Emre Akbas. MultiPoseNet: Fast Multi-Person Pose Estimation using Pose Residual Network. In ECCV, 2018. [arxiv](https://arxiv.org/abs/1807.04067)

This repo includes PRN (pose residual network) module introduced in Section 3.2 of the  paper.

## Getting Started
We have tested our method on [Coco Dataset](http://cocodataset.org)

### Prerequisites

```
python
pytorch
numpy
tqdm
pycocotools
progress
scikit-image
```

### Installing

1. Clone this repository 
`git clone https://github.com/salihkaragoz/pose-residual-network-pytorch.git`

2. Install [Pytorch](https://pytorch.org/)

3. `pip install -r src/requirements.txt`

4. To download COCO dataset train2017 and val2017 annotations run: `bash data/coco.sh`. (data size: ~240Mb)

## Training

`python train.py`

For more options look at opt.py

## Results
Results on COCO val2017 Ground Truth data.

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.880
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.968
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.908
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.870
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.898
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.904
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.974
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.920
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.889
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.931
```

## License

## Citation
If you find this code useful for your research, please consider citing the following paper:
```
@Inproceedings{kocabas18prn,
  Title          = {Multi{P}ose{N}et: Fast Multi-Person Pose Estimation using Pose Residual Network},
  Author         = {Kocabas, Muhammed and Karagoz, Salih and Akbas, Emre},
  Booktitle      = {European Conference on Computer Vision (ECCV)},
  Year           = {2018}
}
```
