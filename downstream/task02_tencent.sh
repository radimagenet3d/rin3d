
export CUDA_VISIBLE_DEVICES=0
fastestimator run segmentation.py --data_path /data/data/rin3d_sample/Task02 \
                  --csv_path /data/rin3d/downstream/data/task02 \
                  --output_dir /data/data/rin3d_sample/Task02/output_tencent \
                  --pretrain tencent \
                  --weight_path /data/data/rin3d_sample/weights/pretrain/resnet_50.pth \
                  --batch_size 1 \
                  --epochs 10 \
                  --init_lr 1e-4 \
                  --log_steps 10 \
                  --train_steps_per_epoch 50