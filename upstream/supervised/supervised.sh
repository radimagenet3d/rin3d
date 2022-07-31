export CUDA_VISIBLE_DEVICES=0
fastestimator run classification.py --data_path /data/data/rin3d_sample/upstream/image \
                  --csv_path /data/rin3d/upstream/data \
                  --output_dir /data/data/rin3d_sample/upstream/supervised \
                  --batch_size 2 \
                  --epochs 20 \
                  --num_class 7 \
                  --init_lr 1e-4 \
                  --log_steps 20 \
                  --train_steps_per_epoch 100