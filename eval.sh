set -x
# change root to your path of dataset root.
data_root=../argo1_data/
p1_root=../p1_data/
# the version directory of the experiment name used in training.
ckpt_version=./ckpts/version_6191823/
pwd

python eval.py \
       --data_root $data_root --p1_root $p1_root \
       --ckpt_dir $ckpt_version \
       --refine_num 5 --refine_radius -1 \
       --embed_dim 64 \
