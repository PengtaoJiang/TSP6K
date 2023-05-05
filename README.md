# Traffic Scene Parsing through the TSP6K Dataset
The Official PyTorch code for the proposed segmentation network in [TSP6K dataset](https://arxiv.org/pdf/2303.02835.pdf). 
Code is implemented based on an open source semantic segmentation toolbox, [MMsegmentation](https://github.com/open-mmlab/mmsegmentation).

## Installation 
Please follow the installation instructions in [mmsegmentation]().

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
Download the dataset from [this link]() and put them into ```/data/TSP6K/```.


## Training 
Train SegNext with the proposed Detail Refining Decoder using 
the following command
```
bash tools/dist_train.sh \
configs/tsp6k/segnext_base_1024x1024_160k_tsp6k_msaspp_rrm_5tokens_12heads.py \
8 --auto-resume  
```

## Evaluation
Evaluate the segmentation model with the following command
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
@article{jiang2023traffic,
  title={Traffic Scene Parsing through the TSP6K Dataset},
  author={Jiang, Peng-Tao and Yang, Yuqi and Cao, Yang and Hou, Qibin and Cheng, Ming-Ming and Shen, Chunhua},
  journal={arXiv preprint arXiv:2303.02835},
  year={2023}
}
```