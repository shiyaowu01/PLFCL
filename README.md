# PLFCL: Part-Aware Local Feature Contrastive Learning for 3D Hand Pose Estimation

**Shiyao Wu, Zhisheng Han, Panneerselvam John, Lu Liu, Haiping Ma, Rajeev Raman, Zheheng Jiang**

This repository contains the PLFCL research code for estimating 3D hand poses from depth images and point clouds. The proposed method uses **mesh-to-joint contrastive learning** to align mesh-aware and joint-aware local features with shared hand-part semantics, providing complementary mesh information for occluded joints.

The framework combines joint-wise feature extraction, local joint refinement, and contrastive learning in an **aggregated depth-point (ADP) local feature space**. The manuscript also describes self-supervised mesh generation for datasets without MANO parameter annotations.

## 📰 Updates

- **11/2026**: Align network and contrastive learning names with the PLFCL manuscript; update training entry points and add legacy checkpoint key migration.

## ⚙️ Requirements

The supplied environment targets **Linux with an NVIDIA GPU and CUDA**. The environment export in [`plfcl.yaml`](plfcl.yaml) includes Python 3.6.9 and PyTorch 1.9.0-era dependencies.

```bash
git clone https://github.com/shiyaowu01/PLFCL.git
cd PLFCL
conda env create -f plfcl.yaml
conda activate plfcl
```

The environment file contains platform-specific package builds and an original-machine `prefix`; adjust these entries to match your installation. The code additionally imports `einops`, which is not listed in the export. For the exported Python 3.6 environment:

```bash
pip install einops==0.3.2
```

Build the PointNet++ CUDA extension using the instructions in [`pointnet2/README.md`](pointnet2/README.md):

```bash
cd pointnet2
python setup.py install
cd ..
```

Use a CUDA toolkit and compiler compatible with your PyTorch installation. Rebuild the extension locally rather than relying on existing build artifacts.

## 📦 Set Up Dataset

### NYU Hand Pose Dataset

Download the NYU Hand Pose Dataset and prepare the following layout. The path is configurable through `--dataset_path` for training and `--test_path` for evaluation.

```text
/path/to/nyu/dataset/
|-- train/
|   |-- joint_data.mat
|   |-- depth_1_0000001.png
|   |-- ...
|-- test/
|   |-- joint_data.mat
|   |-- depth_1_0000001.png
|   |-- ...
|-- center_train_1_refined.txt
|-- center_test_1_refined.txt
```

The current NYU loader uses camera view 1 and defaults to refined crop centers. Supply the two refined-center files before running the provided commands. Their rows must correspond to the dataset frames and contain the 3D center coordinates expected by [`dataloader/loader.py`](dataloader/loader.py).

The manuscript evaluates PLFCL on NYU, ICVL, MSRA, and HO3D. The current training entry point contains NYU and DexYCB branches; it does not provide dedicated training commands for all manuscript datasets.

## ✋ Set Up MANO

Obtain the MANO model files under the model provider's terms. The current implementation in [`mano.py`](mano.py) expects the right-hand model at:

```text
common/utils/manopth/mano/models/
|-- MANO_RIGHT.pkl
```

Create this directory relative to the repository root, or update the `mano_root` path in `MANO.get_layer()` to your local model directory. MANO model files are separate from this repository's source-code license.

## 🚆 Training

Train on NYU with the current entry point:

```bash
python train_plfcl.py \
    --gpu 0 \
    --dataset nyu \
    --dataset_path /path/to/nyu/dataset \
    --model_name plfcl \
    --iters 500
```

Important options include:

| Argument | Default | Description |
| --- | --- | --- |
| `--batchSize` | `128` | Training batch size |
| `--nepoch` | `35` | Number of training epochs |
| `--learning_rate` | `0.001` | Initial learning rate |
| `--save_root_dir` | `results` | Output root directory |
| `--model` | Empty | Model checkpoint to load from the run directory |
| `--optimizer` | Empty | Optimizer checkpoint to load from the run directory |
| `--start_epoch` | `0` | Starting epoch when resuming |

These are the current code defaults. They should be distinguished from the experimental settings reported in the manuscript.

For NYU with `--iters 500`, outputs are written to:

```text
results/nyu_plfcl_500iters_com/
|-- train.log
|-- best_model.pth
|-- best_optimizer.pth
```

Load an existing run by setting `--model best_model.pth`, `--optimizer best_optimizer.pth`, and the appropriate `--start_epoch`.

## 📊 Evaluation

The repository includes the inherited multi-hypothesis evaluation entry point [`eval_mh.py`](eval_mh.py). Its command-line configuration is:

```bash
python eval_mh.py \
    --gpu 0 \
    --dataset nyu \
    --test_path /path/to/nyu/dataset \
    --model_name plfcl \
    --model best_model.pth \
    --iters 500 \
    --test_step 1 \
    --test_hypo 2
```

Place your checkpoint in `results/nyu_plfcl_500iters_com/`, or pass an absolute checkpoint path to `--model`. No pretrained checkpoint download link is currently provided here.

**Current evaluation status:** `eval_mh.py` calls the legacy diffusion-based `PLFCL.sample()` path. That method currently unpacks six values from `encode()`, which returns seven. It requires alignment with the current refinement path before this command can complete. The naming and interface changes have not been validated with an end-to-end GPU run.

## 🗂️ Code Structure

```text
PLFCL/
|-- network_plfcl.py              # PLFCL network and joint refinement modules
|-- train_plfcl.py                # Training and validation entry point
|-- eval_mh.py                    # Legacy multi-hypothesis evaluation entry point
|-- demo.py                       # Visualization/debugging entry point
|-- plfcl.yaml                    # Exported Conda environment
|-- mano.py                       # MANO model wrapper
|-- dataloader/                   # Dataset loading and preprocessing
|-- model/
|   |-- mesh2joint_contrastive.py  # Contrastive loss utilities
|   |-- parallel.py               # Parallel loss and sampling dispatch
|   |-- transformer.py            # Transformer module
|-- convNeXT/                     # Image feature backbone
|-- pointnet2/                    # PointNet++ CUDA operations
```

The network class is `PLFCL`; its contrastive refinement wrapper is `Mesh2JointContrastive`. Existing checkpoints using `localconstrastive.*` are automatically migrated to `mesh2joint_contrastive.*` when loaded. New checkpoints use the updated names.

## 📚 Citation

If you use this work, please cite the PLFCL manuscript:

```bibtex
@misc{wu2026plfcl,
  title={PLFCL: Part-Aware Local Feature Contrastive Learning for 3D Hand Pose Estimation},
  author={Wu, Shiyao and Han, Zhisheng and John, Panneerselvam and Liu, Lu and Ma, Haiping and Raman, Rajeev and Jiang, Zheheng},
  year={2026},
  note={Manuscript}
}
```

This manuscript entry can be replaced with the final publication metadata when available.

## 📄 License

The repository includes an [MIT license](LICENSE). Bundled dependencies retain their own licenses, and MANO model files and datasets remain subject to their providers' terms.

## 🙏 Acknowledgements

This code builds on the HandDiff codebase and the image-point cloud framework from [IPNet](https://github.com/PengfeiRen96/IPNet). It also uses the bundled PointNet++, ConvNeXt, and manopth components. We thank the authors for sharing their work.

The original HandDiff reference is:

```bibtex
@inproceedings{cheng2024handdiff,
  title={HandDiff: 3D Hand Pose Estimation with Diffusion on Image-Point Cloud},
  author={Cheng, Wencan and Tang, Hao and Van Gool, Luc and Ko, Jong Hwan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
