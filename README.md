# Pose Residual Network

This is the code for the paper

Muhammed Kocabas, Salih Karagoz, Emre Akbas. MultiPoseNet: Fast Multi-Person Pose Estimation using Pose Residual Network. In ECCV, 2018. https://arxiv.org/abs/1807.04067

We just released only Pose Residual Network part of the paper. We are planing to share whole code after finishing.


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

1. Clone this repository ```git clone https://github.com/salihkaragoz/pose-residual-network-pytorch.git```
2. Install Pytorch in accordance with your system. [Pytorch](https://pytorch.org/)

3. ```pip install -r src/requirements.txt```
4. To download COCO dataset Train and Validation 2017 Annotations run the bash. (Datasize: ~240Mb)
```bash data/coco.sh```


## Training

``` python train.py ```

For more options look at opt.py

## Results
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

