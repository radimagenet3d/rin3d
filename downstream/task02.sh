
export CUDA_VISIBLE_DEVICES=0
fastestimator run segmentation.py --data_path /data/data/rin3d_sample/Task02 \
                  --csv_path /data/rin3d/downstream/data/task02 \
                  --output_dir /data/data/rin3d_sample/Task02/output \
                  --pretrain True \
                  --batch_size 1 \
                  --epochs 10 \
                  --init_lr 1e-4 \
                  --log_steps 10 \
                  --train_steps_per_epoch 50