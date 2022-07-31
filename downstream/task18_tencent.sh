
export CUDA_VISIBLE_DEVICES=0
fastestimator run classification.py --data_path /data/data/rin3d_sample/Task18/image \
                  --csv_path /data/rin3d/downstream/data/task18 \
                  --output_dir /data/data/rin3d_sample/Task18/output_tencent \
                  --pretrain tencent \
                  --weight_path /data/data/rin3d_sample/weights/pretrain/resnet_50.pth \
                  --batch_size 2 \
                  --epochs 10 \
                  --init_lr 1e-4 \
                  --log_steps 20 \
                  --train_steps_per_epoch None