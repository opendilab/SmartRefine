# SmartRefine: A Scenario-Adaptive Refinement Framework for Efficient Motion Prediction

**_Fast Takeaway_:** We introduce a novel approach to refining motion predictions in autonomous vehicle navigation with minimal additional computation by leveraging scenario-specific properties and adaptive refinement iterations.
![pipeline](assets/pipeline.png)
> Yang Zhou\* , [Hao Shao](http://hao-shao.com/)\* , [Letian Wang](https://letianwang0.wixsite.com/myhome) , [Steven L. Waslander](https://www.trailab.utias.utoronto.ca/stevenwaslander) , [Hongsheng Li](http://www.ee.cuhk.edu.hk/~hsli/) , [Yu Liu](https://liuyu.us/)$^\dagger$.

This repository contains the official implementation of [SmartRefine: A Scenario-Adaptive Refinement Framework for Efficient Motion Prediction](https://arxiv.org/abs/2403.11492) published in _CVPR 2024_.

If you have any concern, feel free to contact: kmzy at hnu.edu.cn or kmzy99 at gmail.com.

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fopendilab%2FSmartRefine%2F&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)

## News

- `[04 Jun., 2024]` We gave a talk at [自动驾驶之心](https://www.zdjszx.com/), the slides can be seen [here](https://github.com/opendilab/SmartRefine/blob/main/SmartRefine_talk.pdf).

- `[22 Mar., 2024]` We released our code for [Argoverse 1](https://github.com/argoverse/argoverse-api). Give it a try!
- `[18 Mar., 2024]` We released our SmartRefine paper on [<u>_arXiv_</u>](https://arxiv.org/abs/2403.11492).
- `[27 Feb., 2024]` Our SmartRefine was accepted by _CVPR 2024_.

## Getting Started
1\. Clone this repository:
```bash
cd $YOUR_WORK_SPACE
git clone https://github.com/opendilab/SmartRefine.git
cd SmartRefine
```
2\. Install the dependencies:
```bash
pip install -r requirements.txt
cd ../
```
You can selectively configure the environment in your favorite way.

3\. Install the [Argoverse-API](https://github.com/argoverse/argoverse-api?tab=readme-ov-file#installation) and download the [Argoverse Motion Forecasting Dataset v1.1](https://www.argoverse.org/av1.html) following the corresponding User Guide under `$YOUR_WORK_SPACE`. Here is an example of extracting the downloaded Argoverse data:

```bash
cd $YOUR_WORK_SPACE
mkdir argo1_data
tar xzvf forecasting_train_v1.1.tar.gz -C ./argo1_data
tar xzvf forecasting_val_v1.1.tar.gz -C ./argo1_data
```

4\. Download the prediction backbone's outputs at [Here](https://openxlab.org.cn/datasets/kmzy99/SmartRefine/tree/main/prediction_data) and extract:

```bash
cd $YOUR_WORK_SPACE
mkdir p1_data
unzip hivt_p1_data.zip -d ./p1_data
```

The final fles inside `$YOUR_WORK_SPACE` should be organized as follows:

```
$YOUR_WORK_SPACE
├── argoverse-api
├── argo1_data
    ├── train
    │   ├── data
    │   │   ├── 1.csv
    │   │   ├── 2.csv
    │   │   └── ...
    └── val
        ├── data
        │   ├── 1.csv
        │   ├── 2.csv
        │   └── ...
        └── Argoverse-Terms_of_Use.txt
├── p1_data
    ├── train
    │   ├── 1.pkl
    │   ├── 2.pkl
    │   └── ...
    └── val
        ├── 1.pkl
        ├── 2.pkl
        └── ...
├── SmartRefine
```
Here, each pickle file inside p1_data contains the backbone model's outputs: predicted trajectories with a shape of $[K, T, 2]$ and trajectory features shaped as $[K, -1]$, where $K$ is the number of modalities and $T$ is the trajectory length.

5\. **[Optional]** Generate your own model's prediction outputs.

As mentioned in our paper, SmartRefine is designed to be decoupled from the primary prediction model backbone, and only requires a generic interface to the model backbone (predicted trajectories and trajectory features). Therefore, we present a script `eval_store.py` as an example to show how to store the backbone's outputs. The main idea is to store predicted trajectories with a key of 'traj' and trajectory features as 'embed' into a dictionary.

## Training
You can train the model on a single GPU or multiple GPUs to accelerate the training process:

```bash
cd $YOUR_WORK_SPACE
cd SmartRefine
bash train.sh
```

You can change your training setting. The default `train.sh` looks like as follows:
```bash
set -x
# change root to your path of dataset root.
data_root=../argo1_data/
# change p1_root to your path of prediction outputs root.
p1_root=../p1_data/
# experiment name used for logging.
exp=smartref_hivt_argo1
# device number.
ngpus=1
pwd

python train.py \
       --data_root $data_root --p1_root $p1_root --exp $exp \
       --train_batch_size 32 --val_batch_size 32 \
       --gpus $ngpus --embed_dim 64 --refine_num 5 --seg_num 2 \
       --refine_radius -1 --r_lo 2 --r_hi 10 \
```

**_Note_**: The first training epoch will take longer because it preprocess the data at the same time. The regular training time per epoch is around 20~40 minutes varied by different hardware.

The training process will be saved in `$exp/lightning_logs/` automatically. To monitor it:
```bash
cd $exp
tensorboard --logdir lightning_logs/
```

## Evaluation
To evaluate the model performance:
```bash
cd $YOUR_WORK_SPACE
cd SmartRefine
bash eval.sh
```

## Results
### Tabular Results
The expected performance is:
| Methods      | minFDE | minADE | MR   |
| ------------ | ------ | ------ | ---- |
| HiVT         | 0.969   | 0.661   | 0.092 |
| HiVT w/ Ours | 0.913   | 0.646   | 0.083 |
### Visualization Results
The dark blue arrows are multi-nodal predictions of the agent by model and the pink arrow is the ground truth future trajectory respectively. The shortest trajectory gets more aligned toward the ground truth direction, and the trajectory closest to the ground truth gets closer after refinement.
![vis](assets/visualization.png)

## Citation
If you find our repo or paper useful, please cite us as:

```bibtex
@misc{zhou2024smartrefine,
      title={SmartRefine: A Scenario-Adaptive Refinement Framework for Efficient Motion Prediction}, 
      author={Yang Zhou and Hao Shao and Letian Wang and Steven L. Waslander and Hongsheng Li and Yu Liu},
      year={2024},
      eprint={2403.11492},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements

This implementation is based on code from other repositories.
- [HiVT](https://github.com/ZikangZhou/HiVT)
- [LMDrive](https://github.com/opendilab/LMDrive)
- [Forecast-MAE](https://github.com/jchengai/forecast-mae)

## License

All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).