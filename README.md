# Pose Residual Network

This is the code for the paper

Muhammed Kocabas, Salih Karagoz, Emre Akbas. MultiPoseNet: Fast Multi-Person Pose Estimation using Pose Residual Network. In ECCV, 2018. [arxiv](https://arxiv.org/abs/1807.04067)

This is the code for the PRN (pose residual network) module introduced in Section 3.2 of the  paper.



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
```git clone https://github.com/salihkaragoz/pose-residual-network-pytorch.git```

2. Install Pytorch in accordance with your system. [Pytorch](https://pytorch.org/)

3. ```pip install -r src/requirements.txt```

4. To download COCO dataset train2017 and val2017 annotations run: `bash data/coco.sh`. (data size: ~240Mb)

## Training

``` python train.py ```

For more options look at opt.py

## Results
We have just published only the results on COCO Val Groun Truth.

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.877
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.968
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.899
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.872
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.894
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.902
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.973
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.919
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.888
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.929
```


## Authors

## License

## Acknowledgments

## Citation
If you find this code useful for your research, please consider citing the following paper:
```
@Inproceedings{kocabas18prn,
  Title          = {Multi{P}ose{N}et: Fast Multi-Person Pose Estimation using Pose Residual Network},
  Author         = {Kocabas, Muhammed and Karagoz, Salih and Akbas, Emre},
  Booktitle      = {European Conference on Computer Vision (ECCV)},
  Year           = {2017}
}
```
