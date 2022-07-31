
export CUDA_VISIBLE_DEVICES=0
fastestimator run classification.py --data_path /data/data/rin3d_sample/Task18/image \
                  --csv_path /data/rin3d/downstream/data/task18 \
                  --output_dir /data/data/rin3d_sample/Task18/output_rin3d \
                  --pretrain rin3d \
                  --weight_path /data/data/rin3d_sample/Task02/output_scratch/backbone_best_Dice.pt \
                  --batch_size 2 \
                  --epochs 10 \
                  --init_lr 1e-4 \
                  --log_steps 20 \
                  --train_steps_per_epoch None