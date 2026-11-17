# PLFCL: Part-Aware Local Feature Contrastive Learning for 3D Hand Pose Estimation

**Shiyao Wu, Zhisheng Han, Panneerselvam John, Lu Liu, Haiping Ma, Rajeev Raman, Zheheng Jiang**

**IEEE Transactions on Multimedia (TMM)**

PLFCL estimates 3D hand poses from depth images and point clouds. Mesh-to-joint contrastive learning aligns local mesh and joint features with shared hand-part semantics to improve pose estimation under occlusion.

## Installation

The provided environment targets Linux with an NVIDIA GPU and CUDA.

```bash
git clone https://github.com/shiyaowu01/PLFCL.git
cd PLFCL
conda env create -f plfcl.yaml
conda activate plfcl
pip install einops==0.3.2
cd pointnet2
python setup.py install
cd ..
```

Adjust platform-specific packages and the `prefix` in `plfcl.yaml` for your machine. Use a CUDA toolkit compatible with PyTorch.

## Data Preparation

Prepare the NYU Hand Pose Dataset as follows:

```text
/path/to/nyu/dataset/
|-- train/                         # joint_data.mat and depth_1_*.png
|-- test/                          # joint_data.mat and depth_1_*.png
|-- center_train_1_refined.txt
|-- center_test_1_refined.txt
```

The loader uses camera view 1 and refined crop centers. Obtain the MANO right-hand model separately and place `MANO_RIGHT.pkl` under `common/utils/manopth/mano/models/`, or update the path in `mano.py`.

## Training

```bash
python train_plfcl.py \
    --gpu 0 \
    --dataset nyu \
    --dataset_path /path/to/nyu/dataset \
    --model_name plfcl \
    --iters 500
```

Validation runs after each epoch. Logs and the best model/optimizer checkpoints are saved to `results/nyu_plfcl_500iters_com/`. A standalone evaluation script and pretrained checkpoint download are not currently provided.

## Citation

```bibtex
@article{wu2026plfcl,
  title={PLFCL: Part-Aware Local Feature Contrastive Learning for 3D Hand Pose Estimation},
  author={Wu, Shiyao and Han, Zhisheng and John, Panneerselvam and Liu, Lu and Ma, Haiping and Raman, Rajeev and Jiang, Zheheng},
  journal={IEEE Transactions on Multimedia},
  year={2026}
}
```

## Acknowledgements

This code builds on HandDiff and [IPNet](https://github.com/PengfeiRen96/IPNet), and uses PointNet++, ConvNeXt, and manopth. We thank the authors for sharing their work.

## License

See [LICENSE](LICENSE). Dependencies, datasets, and MANO models retain their respective licenses and terms.
