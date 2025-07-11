# [CVPR2024] Traffic Scene Parsing through the TSP6K Dataset

<!-- <div align="center">
<img src=https://github.com/PengtaoJiang/TSP6K/blob/main/tsp6k_logo.png width=400 height=120/>
</div> -->
<div align="center"><video src="https://github.com/PengtaoJiang/TSP6K/assets/23328456/217770e6-5d0b-4a3d-a709-ebe752857c85" width="700" muted="false"></video></div>

The dataset and code in [TSP6K dataset](https://arxiv.org/pdf/2303.02835.pdf). 
Code is implemented using an open-source semantic segmentation toolbox, [MMsegmentation](https://github.com/open-mmlab/mmsegmentation).

## Installation 
Please follow the installation instructions in [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation). 
In our environment, we use the following versions of different packages.
```
mmsegmentation==0.20.2
mmcv-full=1.4.0
```


Install the mmseg lib first 
```
git clone https://github.com/PengtaoJiang/TSP6K.git
cd TSP6K/
pip install -v -e .
```
If you want to evaluate the iIoU score, please install the cityscapesscript lib
```
cd mmseg/datasets/cityscapesscripts/
python setup.py build install
```

## Dataset Preparation
Download the dataset from [this link(Google Drive)](https://drive.google.com/file/d/1yDCL8vLOYcRuInCPUij3_pmq9yPk7KQj/view?usp=sharing) or [this link(jianguoyun)](https://www.jianguoyun.com/p/DZVO0kMQ2tnWChjun6MFIAA)(password: Wi9qFT) or [this link(baidu disk)](https://pan.baidu.com/s/15hL4WcvngN8GFhpU59ruOw)(password: jzra) and put them into ```/data/TSP6K/```.
```
data
├── TSP6K
│   ├── image
│   ├── label
│   ├── split
```
You can also download the COCO-style instance bounding box annotations from this [link](https://drive.google.com/file/d/1bhTQPZ_HS9DttmiMzeCWyjWhAD5Fg9q8/view?usp=drive_link).

## Training 
Train SegNext with the proposed Detail Refining Decoder using 
the following command
```
bash tools/dist_train.sh \
configs/tsp6k/segnext_base_1024x1024_160k_tsp6k_msaspp_rrm_5tokens_12heads.py \
8 --auto-resume  
```

## Evaluation

### Results and models

| Method | Backbone | Crop Size |Lr Sche. |  val mIoU (ms) | val iIoU (ms) | config | model |
| :----- |:-----:   |:-----:    |:---:    |:---:  |:---: |:---:   |:---:  |
| SegNext+DRD | MSCAN-B  | 1024x1024  | 160000  | 75.8 | 58.4 | [config](https://github.com/PengtaoJiang/TSP6K/blob/main/configs/tsp6k/segnext_base_1024x1024_160k_tsp6k_msaspp_rrm_5tokens_12heads.py)  | [model](https://www.jianguoyun.com/p/DfREB7IQ2tnWChjhn6MFIAA) |
| SegNext+DRD | MSCAN-L  | 1024x1024  | 160000  | 76.2 | 58.9 | [config](https://github.com/PengtaoJiang/TSP6K/blob/main/configs/tsp6k/segnext_large_1024x1024_160k_tsp6k_msaspp_rrm.py)  | [model](https://www.jianguoyun.com/p/Dca5qT8Q2tnWChjtn6MFIAA) | 

We provide the pre-trained segmentation models above. You can download them and directly evaluate them by
```
bash tools/dist_test.sh \
    configs/tsp6k/segnext_base_1024x1024_160k_tsp6k_msaspp_rrm_5tokens_12heads.py \
    ./work_dirs/segnext_base_1024x1024_160k_tsp6k_msaspp_rrm_5tokens_12heads/latest.pth \
    8 --out ./work_dirs/segnext_base_1024x1024_160k_tsp6k_msaspp_rrm_5tokens_12heads/results.pkl \
    --aug-test --eval mIoU  
```
Evaluate the segmentation model using the iIoU metric by 
```
bash tools/dist_test.sh \
    configs/tsp6k/segnext_base_1024x1024_160k_tsp6k_msaspp_rrm_5tokens_12heads.py \
    ./work_dirs/segnext_base_1024x1024_160k_tsp6k_msaspp_rrm_5tokens_12heads/latest.pth \
    8 --out ./work_dirs/segnext_base_1024x1024_160k_tsp6k_msaspp_rrm_5tokens_12heads/results.pkl \
    --aug-test --eval cityscapes  
```

## Citation
If you find the proposed TSP6K dataset and segmentation network are useful for your research, please cite
```
@inproceedings{jiang2024traffic,
  title={Traffic Scene Parsing through the TSP6K Dataset},
  author={Jiang, Peng-Tao and Yang, Yuqi and Cao, Yang and Hou, Qibin and Cheng, Ming-Ming and Shen, Chunhua},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  year={2024}
}
```
