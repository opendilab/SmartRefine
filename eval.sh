set -x
# change root to your path of dataset root.
data_root=../argo1_data/
# the version directory of the experiment name used in training.
ckpt_version=smartref_hivt_argo1/lightning_logs/version_xxx/
pwd

python eval.py \
       --data_root $data_root --ckpt-path $ckpt_version \
       --refine_num 5 --refine_radius -1 \
