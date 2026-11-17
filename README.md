# PLFCL: Part-Aware Local Feature Contrastive Learning for 3D Hand Pose Estimation

Shiyao Wu, Zhisheng Han, Panneerselvam John, Lu Liu, Haiping Ma, Rajeev Raman, Zheheng Jiang

This project's naming follows the PLFCL manuscript:

- **PLFCL**: Part-Aware Local Feature Contrastive Learning.
- **Mesh-to-joint contrastive learning**: the contrastive learning strategy described in Section III-B (Mesh-to-joint contrast).
- Network module: `network_plfcl.py`; network class: `PLFCL`.
- Contrastive refinement class: `Mesh2JointContrastive`.
- Contrastive loss module: `model/mesh2joint_contrastive.py`; loss functions: `mesh2joint_intra_contrastive_loss` and `mesh2joint_inter_contrastive_loss`. The intra/inter suffixes distinguish the existing implementations; they are code identifiers, not additional method names from the paper.

Training, evaluation and demo scripts default to `--model_name plfcl`.
The Conda environment definition is `plfcl.yaml`.
The contrastive module is registered as `mesh2joint_contrastive`. Loading a checkpoint automatically migrates the old `localconstrastive.*` keys; conflicting old/new keys are rejected. Other parameter keys are unchanged.
The training entry point is `train_plfcl.py`. Local refinement uses `JointRefinement` and `JointRefinementGraph`, and the combined training loss is named `total_loss`.
Multi-GPU scripts use `PLFCLDataParallel` to dispatch prediction, loss and sampling calls. Saved checkpoints omit the wrapper's `module.` prefix so they can be loaded on a single GPU.
The inherited `DiffusionPoint`, `LocalFolding` and `LocalFoldingGraph` components retain their functional names; they are separate from the contrastive refinement path.
New runs use `results/nyu_plfcl_500iters_com`; copy existing weights there or specify `--model`.
These naming changes do not modify the model architecture or loss computation.

## Original HandDiff reference

Wencan Cheng, Hao Tang, Luc Van Gool and Jong Hwan Ko

IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024
*Highlight

---
## Prerequisities
Our model is trained and tested under:
* Python 3.6.9
* NVIDIA GPU + CUDA CuDNN
* PyTorch (torch == 1.9.0)
* scipy
* tqdm
* Pillow
* yaml
* json
* cv2
* pycocotools

1. Prepare dataset 

    please download the NYU Hand dataset

2. Install PointNet++ CUDA operations

    follow the instructions in the './pointnet2' for installation 

3. Evaluate

    set the "--test_path" paramter in the ```test_nyu.sh ``` as the path saved the generated testing set

    execute ``` sh test_nyu.sh```

    we provided the pre-trained models ('./results/nyu_plfcl_500iters_com/best_model.pth') for NYU

4. If a new training process is needed, please execute the following instructions after step 1 and 2 are completed

    set the "--dataset_path" paramter in the ```train_nyu.sh ``` as the path saved the generated traning and testing set respectively

    execute ``` sh train_nyu.sh```

The original HandDiff reference is retained below for attribution:
```
@inproceedings{cheng2024handdiff,
  title={HandDiff: 3D Hand Pose Estimation with Diffusion on Image-Point Cloud},
  author={Cheng, Wencan and Tang, Hao and Van Gool, Luc and Ko, Jong Hwan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

## Acknowledgement

We thank [repo](https://github.com/PengfeiRen96/IPNet) for the image-point cloud framework.


