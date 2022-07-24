import os
import pdb

import nibabel
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom

import fastestimator as fe
from fastestimator.dataset import CSVDataset
from fastestimator.op.numpyop.numpyop import Delete, NumpyOp
from fastestimator.op.tensorop import Dice
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver, CSVLogger
from fastestimator.trace.metric import Dice as DiceScore


class ReadImage(NumpyOp):
    def __init__(self, parent_path, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.parent_path = parent_path

    def forward(self, data, state):
        file_path = os.path.join(self.parent_path, data)
        img_data = nibabel.load(file_path)
        img_data = img_data.get_fdata()
        return np.int32(img_data)


class Resize3D(NumpyOp):
    def forward(self, data, state):
        img_data = data
        z, x, y = img_data.shape
        scale_z, scale_x, scale_y = 128 / z, 128 / x, 128 / y
        img_data = zoom(img_data, (scale_z, scale_x, scale_y))
        return img_data


class Rescale(NumpyOp):
    def forward(self, data, state):
        img_data = data
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
        return np.float32(img_data)


class AddChannel(NumpyOp):
    def __init__(self, num_channel, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.num_channel = num_channel

    def forward(self, data, state):
        img_data = data
        img_data = np.stack([img_data for _ in range(self.num_channel)], axis=0)
        return img_data


class R50TorchVideoEncoder(nn.Module):
    def __init__(self, pretrain=False):
        super().__init__()
        res50_layers = list(torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=pretrain).children())
        self.res50_in_C1 = nn.Sequential(*list(res50_layers[0][0].children())[:-1])
        self.C1_pool = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                    stride=(1, 2, 2),
                                    padding=[0, 1, 1],
                                    dilation=1,
                                    ceil_mode=False)
        self.res50_C1_C2 = nn.Sequential(res50_layers[0][1])
        self.res50_C2_C3 = nn.Sequential(res50_layers[0][2])
        self.res50_C3_C4 = nn.Sequential(res50_layers[0][3])
        self.res50_C4_C5 = nn.Sequential(res50_layers[0][4])

    def forward(self, x):
        C1 = self.res50_in_C1(x)
        C2 = self.C1_pool(C1)  # move the maxpool from C1 to C2 to stay consistent with Unet design
        C2 = self.res50_C1_C2(C2)
        C3 = self.res50_C2_C3(C2)
        C4 = self.res50_C3_C4(C3)
        C5 = self.res50_C4_C5(C4)
        return C1, C2, C3, C4, C5


class R50TorchVideoDecoder(nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()
        self.dec_C5 = DecBlock(2048, 1536, 512)
        self.dec_C4 = DecBlock(512, 768, 256)
        self.dec_C3 = DecBlock(256, 384, 128)
        self.dec_C2 = DecBlock(128, 128, 64)
        self.dec_C1 = nn.Sequential(nn.Upsample(scale_factor=(1.0, 2.0, 2.0)),
                                    nn.Conv3d(64, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(32, num_channels, kernel_size=(1, 1, 1), padding=(0, 0, 0)))

    def forward(self, C1, C2, C3, C4, C5):
        D4 = self.dec_C5(C5, C4)
        D3 = self.dec_C4(D4, C3)
        D2 = self.dec_C3(D3, C2)
        D1 = self.dec_C2(D2, C1)
        D0 = self.dec_C1(D1)
        mask_out = torch.sigmoid(D0)
        return mask_out


class DecBlock(nn.Module):
    """Decoder Block"""
    def __init__(self, upsample_in_ch, conv_in_ch, out_ch):
        super().__init__()
        self.upsample_conv = nn.Sequential(nn.Upsample(scale_factor=(1.0, 2.0, 2.0)),
                                           nn.Conv3d(upsample_in_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                                           nn.ReLU(inplace=True))

        self.conv_layers = nn.Sequential(nn.Conv3d(conv_in_ch, out_ch, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
                                         nn.ReLU(inplace=True),
                                         nn.Conv3d(out_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                                         nn.ReLU(inplace=True))

    def forward(self, x_up, x_down):
        x = self.upsample_conv(x_up)
        x = torch.cat([x, x_down], 1)
        x = self.conv_layers(x)
        return x


def get_estimator(data_path,
                  csv_path,
                  output_dir,
                  pretrain=False,
                  batch_size=1,
                  epochs=10,
                  init_lr=1e-4,
                  log_steps=20,
                  train_steps_per_epoch=None):
    ds_train = CSVDataset(os.path.join(csv_path, "train.csv"))
    ds_val = CSVDataset(os.path.join(csv_path, "val.csv"))
    ds_test = CSVDataset(os.path.join(csv_path, "test.csv"))
    pipeline = fe.Pipeline(
        train_data=ds_train,
        eval_data=ds_val,
        test_data=ds_test,
        batch_size=batch_size,
        ops=[
            ReadImage(parent_path=os.path.join(data_path, "image"), inputs="img_filename", outputs="image_data"),
            ReadImage(parent_path=os.path.join(data_path, "label"), inputs="mask_filename", outputs="mask_data"),
            Resize3D(inputs="image_data", outputs="image_data"),
            Resize3D(inputs="mask_data", outputs="mask_data"),
            Rescale(inputs="image_data", outputs="image_data"),
            Rescale(inputs="mask_data", outputs="mask_data"),
            AddChannel(inputs="image_data", outputs="image_data", num_channel=3),
            AddChannel(inputs="mask_data", outputs="mask_data", num_channel=1),
            Delete(keys=("partition", "mask_filename")),
            Delete(keys="img_filename", mode="!test")
        ])
    backbone = fe.build(model_fn=lambda: R50TorchVideoEncoder(pretrain=pretrain),
                        optimizer_fn=lambda x: torch.optim.Adam(x, lr=init_lr),
                        model_name="backbone")
    seg_head = fe.build(model_fn=R50TorchVideoDecoder,
                        optimizer_fn=lambda x: torch.optim.Adam(x, lr=init_lr),
                        model_name="seg_head")
    network = fe.Network(ops=[
        ModelOp(inputs="image_data", model=backbone, outputs=("C1", "C2", "C3", "C4", "C5")),
        ModelOp(inputs=("C1", "C2", "C3", "C4", "C5"), model=seg_head, outputs="mask_pred"),
        Dice(inputs=("mask_pred", "mask_data"), outputs="dice_loss", sample_average=True, negate=True),
        UpdateOp(model=seg_head, loss_name="dice_loss"),
        UpdateOp(model=backbone, loss_name="dice_loss")
    ])
    traces = [
        DiceScore(true_key="mask_data", pred_key="mask_pred"),
        CSVLogger(filename=os.path.join(output_dir, "test_result.csv"), instance_id_key="img_filename", mode="test"),
        BestModelSaver(model=backbone, save_dir=output_dir, metric="Dice", save_best_mode="max"),
        BestModelSaver(model=seg_head, save_dir=output_dir, metric="Dice", save_best_mode="max")
    ]
    est = fe.Estimator(pipeline=pipeline,
                       network=network,
                       epochs=epochs,
                       traces=traces,
                       log_steps=log_steps,
                       train_steps_per_epoch=train_steps_per_epoch)
    return est
