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
