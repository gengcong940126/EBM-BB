# Bounds all around: training energy-based models with bidirectional bounds

Official code for the NeurIPS 2021 paper 
[Bounds all around: training energy-based models with bidirectional bounds](https://arxiv.org/abs/2111.00929).

# Requirements
You can create the environment by running
```
conda env create -f environment.yml
```
# Run model

To train an EBM on a toy dataset, run
```
python ./train/train_toy.py
```
To train an EBM on an image dataset, run
```
 python ./train/train_image.py
```

# Citation
If you find our work helpful to your research, please cite:
```
@article{geng2021bounds,
  title={Bounds all around: training energy-based models with bidirectional bounds},
  author={Geng, Cong and Wang, Jia and Gao, Zhiyong and Frellsen, Jes and Hauberg, S{\o}ren},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```