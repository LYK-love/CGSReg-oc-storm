# ICLR 2026 oc-storm

## Object-Centric World Models from Few-Shot Annotations for Sample-Efficient Reinforcement Learning [[Paper Link]](https://arxiv.org/pdf/2501.16443)

[Weipu Zhang](https://www.weipuzhang.com), [Adam Jelley](https://adamjelley.github.io/), [Trevor McInroe](https://trevormcinroe.github.io/), [Amos Storkey](https://homepages.inf.ed.ac.uk/amos/), [Gang Wang](https://ac.bit.edu.cn/szdw/jsml/mssbyznxtyjs1/224f1108f85a435d9efaaa3dc05fa536.htm)

Work was initiated at the University of Edinburgh and completed at the Beijing Institute of Technology.

[![YouTube](https://img.shields.io/badge/YouTube-Video-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=lGQLdTBY4_Q)
[![Bilibili](https://img.shields.io/badge/Bilibili-Video-00A1D6?logo=bilibili&logoColor=white)](https://www.bilibili.com/video/BV123HyzuEJy)
[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f)](https://oc-storm.weipuzhang.com/)
[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-1f6feb)](https://arxiv.org/pdf/2501.16443)

Watch our video demo above to see the amazing fights played by RL agents!

**TL;DR: oc-storm is an object-centric world-model RL framework that uses few-shot segmentation annotations to improve sample efficiency in Atari and Hollow Knight.**


![oc-storm main figure](assets/main.png)

## Environment installation

### Machine-local data directories

Runtime data and experiment outputs should live outside the git checkout. After
cloning on a new machine, run:

```bash
scripts/setup_links.sh
scripts/check_paths.sh
```

You can also pass the data root non-interactively:

```bash
scripts/setup_links.sh /path/to/oc-storm-data
scripts/check_paths.sh /path/to/oc-storm-data
```

The setup script creates repository-root symlinks for machine-local data
directories. The symlink targets are intentionally not version-controlled
because absolute paths differ across machines.

1. Create conda environment:

    ```bash
    conda create -n oc-storm python=3.12
    ```

2. Activate environment:

    ```bash
    conda activate oc-storm
    ```

3. Install a PyTorch build that matches your NVIDIA driver:

    `pip install torch` does not automatically choose a CUDA build that matches your local driver. Check `nvidia-smi` first, then install a compatible PyTorch wheel explicitly.

    Example for machines whose driver supports CUDA 12.X:

    ```bash
    pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
    ```

    Verify the installation:

    ```bash
    python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
    ```

    If you see a runtime error such as `The NVIDIA driver on your system is too old`, your PyTorch CUDA wheel is newer than the driver supports. Reinstall PyTorch with a compatible CUDA build.

4. Install the remaining Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. Download CUTIE model weights and segmentation masks:

    These assets are not required to run STORM itself. They are only needed for oc-storm, and are not required if you are only interested in running STORM on Hollow Knight.

    ```bash
    bash scripts/download.sh
    ```

    Afterwards, the folder `feature_extractor/cutie/weights` should contain `coco_lvis_h18_itermask.pth` and `cutie-small-mega.pth`, and the project root should contain `segmentation_masks` folder (unless the .tar file was not extracted).

    Or download and extract manually if you prefer: [coco_lvis_h18_itermask.pth](https://github.com/hkchengrex/Cutie/releases/download/v1.0/coco_lvis_h18_itermask.pth) | [cutie-small-mega.pth](https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-small-mega.pth) | [segmentation_masks.tar](https://github.com/weipu-zhang/oc-storm/releases/download/v1.0/segmentation_masks.tar)

    For Atari games, the environment setup is complete after completing this step.

6. For Hollow Knight installation and configuration: [hollow_knight.md](docs/hollow_knight.md)


## Computational requirements

Most of our runs are conducted on 3090/4090, and we recommend using similar devices.

For Atari, a GPU with memory >= 11GB is preferred.


## Train, Evaluate, and Monitor

Train:

```bash
./scripts/train.sh
```

Evaluate:

```bash
./scripts/eval.sh
```

Monitor with TensorBoard:

```bash
./scripts/tensorboard.sh
```

Reproduce the Pong STORM vs oc-storm comparison with `tiny-exp-scheduler`:

```bash
tiny-exp-scheduler run scripts/experiments/pong_storm_vs_oc_storm.commands.txt --cuda-devices auto
```

Full instructions: [docs/reproduce_pong_scheduler.md](docs/reproduce_pong_scheduler.md)

Interactive world-model play (oc-storm and STORM): [docs/play.md](docs/play.md)

Experiment boundary: RL-in-WM and offline world-model regularization use STORM
world models only. The online Pong reproduction is the comparison that trains
both oc-storm and STORM.

Dataset conversion across DIAMOND, Dreamer, and oc-storm/STORM:
[DIAMOND dataset conversion guide](https://github.com/UCDavis-SSL-Lab/diamond/blob/main/docs/dataset-conversion.md)

Stop background training processes (**WARN: Read this first and use at your own risk**):

```bash
./scripts/kill.sh
```

## Citation

```bibtex
@inproceedings{
    zhang2026objectcentric,
    title={Object-Centric World Models from Few-Shot Annotations for Sample-Efficient Reinforcement Learning},
    author={Weipu Zhang and Adam Jelley and Trevor McInroe and Amos Storkey and Gang Wang},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=qmEyJadwHA}
}
```
