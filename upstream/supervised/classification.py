import os
import pdb

import nibabel
import nrrd
import numpy as np
import torch
import torch.nn as nn
from medpy.io import load as load_mha
from scipy.ndimage import zoom

import fastestimator as fe
from fastestimator.dataset import CSVDataset
from fastestimator.op.numpyop.numpyop import Delete, NumpyOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace.io import BestModelSaver
from fastestimator.trace.metric import Accuracy, MCC


class ReadImage(NumpyOp):
    def __init__(self, parent_path, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.parent_path = parent_path

    def forward(self, data, state):
        file_path = os.path.join(self.parent_path, data)
        if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
            img_data = nibabel.load(file_path)
            img_data = img_data.get_fdata()
        elif file_path.endswith('.nrrd'):
            img_data, header = nrrd.read(file_path)
        elif file_path.endswith('.mha'):
            img_data, header = load_mha(file_path)
        elif file_path.endswith('.npy'):
            img_data = np.load(file_path)
        else:
            raise ValueError("unrecognized file {}".format(file_path))
        if len(img_data.shape) == 4:
            img_data = img_data[:, :, :, 0]
        return np.int32(img_data)


class Rescale(NumpyOp):
    def forward(self, data, state):
        img_data = data
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
        return np.float32(img_data)


class Resize3D(NumpyOp):
    def forward(self, data, state):
        img_data = data
        z, x, y = img_data.shape
        scale_z, scale_x, scale_y = 128 / z, 128 / x, 128 / y
        img_data = zoom(img_data, (scale_z, scale_x, scale_y))
        return img_data


class AddChannel(NumpyOp):
    def __init__(self, num_channel, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.num_channel = num_channel

    def forward(self, data, state):
        img_data = data
        img_data = np.stack([img_data for _ in range(self.num_channel)], axis=0)
        return img_data


class R50TorchVideoEncoder(nn.Module):
    def __init__(self, pretrain, weight_path=None):
        super().__init__()
        assert pretrain in [None, "rin3d", "slow"], "pretrain options are None, 'rin3d', or 'slow'"
        if pretrain == "slow":
            res50_layers = list(torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True).children())
        else:
            res50_layers = list(
                torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False).children())
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
        if pretrain == "rin3d":
            assert weight_path, "must provide weights path for rin3d"
            self.load_state_dict(torch.load(weight_path))
            print("loaded weights from {}".format(weight_path))

    def forward(self, x):
        C1 = self.res50_in_C1(x)
        C2 = self.C1_pool(C1)
        C2 = self.res50_C1_C2(C2)
        C3 = self.res50_C2_C3(C2)
        C4 = self.res50_C3_C4(C3)
        C5 = self.res50_C4_C5(C4)
        return C5


class Classifier(nn.Module):
    def __init__(self, num_class=1):
        super().__init__()
        self.linear = nn.Linear(2048, num_class)
        self.num_class = num_class

    def forward(self, x):
        x = nn.functional.adaptive_max_pool3d(x, output_size=(1, 1, 1))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        if self.num_class == 1:
            x = torch.sigmoid(x)
        else:
            x = torch.softmax(x, -1)
        return x


def get_estimator(data_path,
                  csv_path,
                  output_dir,
                  batch_size=2,
                  epochs=20,
                  init_lr=1e-4,
                  log_steps=20,
                  num_class=1,
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
            ReadImage(parent_path=data_path, inputs="filename", outputs="image_data"),
            Resize3D(inputs="image_data", outputs="image_data"),
            Rescale(inputs="image_data", outputs="image_data"),
            AddChannel(inputs="image_data", outputs="image_data", num_channel=3),
            Delete(keys=("filename", "partition", "pathology", "modality", "anatomy"))
        ])
    backbone = fe.build(model_fn=lambda: R50TorchVideoEncoder(pretrain=None),
                        optimizer_fn=lambda x: torch.optim.Adam(x, lr=init_lr),
                        model_name="backbone")
    classifier = fe.build(model_fn=lambda: Classifier(num_class=num_class),
                          optimizer_fn=lambda x: torch.optim.Adam(x, lr=init_lr),
                          model_name="classifier")
    network = fe.Network(ops=[
        ModelOp(inputs="image_data", model=backbone, outputs="feature"),
        ModelOp(inputs="feature", model=classifier, outputs="pred"),
        CrossEntropy(inputs=("pred", "label"), outputs="ce"),
        UpdateOp(model=classifier, loss_name="ce"),
        UpdateOp(model=backbone, loss_name="ce")
    ])
    traces = [
        Accuracy(true_key="label", pred_key="pred"),
        MCC(true_key="label", pred_key="pred"),
        BestModelSaver(model=backbone, save_dir=output_dir, metric="accuracy", save_best_mode="max"),
        BestModelSaver(model=classifier, save_dir=output_dir, metric="accuracy", save_best_mode="max")
    ]
    est = fe.Estimator(pipeline=pipeline,
                       network=network,
                       epochs=epochs,
                       traces=traces,
                       log_steps=log_steps,
                       train_steps_per_epoch=train_steps_per_epoch)
    return est
