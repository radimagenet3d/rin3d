import os
import pdb

import nrrd
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom
from sklearn import metrics

import fastestimator as fe
from fastestimator.dataset import CSVDataset
from fastestimator.op.numpyop.numpyop import Delete, NumpyOp
from fastestimator.op.tensorop.loss import CrossEntropy
from fastestimator.op.tensorop.model import ModelOp, UpdateOp
from fastestimator.trace import Trace
from fastestimator.trace.io import BestModelSaver, CSVLogger


class ReadImage(NumpyOp):
    def __init__(self, parent_path, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.parent_path = parent_path

    def forward(self, data, state):
        file_path = os.path.join(self.parent_path, data)
        img_data, header = nrrd.read(file_path)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               dilation=dilation,
                               padding=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class R50TencentEncoder(nn.Module):
    def __init__(self, block, layers, weight_path=None):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(
            block,
            64,
            layers[0], )
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        assert weight_path, "must provide weight path for tencent model"
        checkpoint = torch.load(weight_path)
        state_dict_new = {}
        for key, val in checkpoint['state_dict'].items():
            newkey = key.replace("module.", "")
            state_dict_new[newkey] = val
        self.load_state_dict(state_dict_new)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        C1 = self.conv1(x)
        C1 = self.bn1(C1)
        C1 = self.relu(C1)
        C2 = self.maxpool(C1)
        C2 = self.layer1(C2)
        C3 = self.layer2(C2)
        C4 = self.layer3(C3)
        C5 = self.layer4(C4)
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


class AUC(Trace):
    def on_epoch_begin(self, data):
        self.y_true, self.y_pred = [], []

    def on_batch_end(self, data):
        y_pred = data["pred"].numpy()
        y_true = data["label"].numpy()
        self.y_pred.append(y_pred)
        self.y_true.append(y_true)
        data.write_per_instance_log("pred", y_pred)
        data.write_per_instance_log("label", y_true)

    def on_epoch_end(self, data):
        y_pred_all = np.concatenate(self.y_pred, axis=0)
        y_true_all = np.concatenate(self.y_true, axis=0)
        num_class = y_pred_all.shape[-1]
        if num_class == 1:
            fpr, tpr, _ = metrics.roc_curve(y_true_all, y_pred_all)
            auc = metrics.auc(fpr, tpr)
            data.write_with_log("auc", auc)
        else:
            aucs = []
            for i in range(num_class):
                y_pred = y_pred_all[:, i]
                y_true = np.where(y_true_all == i, 1, 0)
                fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
                auc = metrics.auc(fpr, tpr)
                data.write_with_log("auc_{}".format(i), auc)
                aucs.append(auc)
            data.write_with_log("auc", np.mean(aucs))


def get_estimator(data_path,
                  csv_path,
                  output_dir,
                  pretrain=None,
                  weight_path=None,
                  batch_size=2,
                  epochs=20,
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
            ReadImage(parent_path=data_path, inputs="filename", outputs="image_data"),
            Resize3D(inputs="image_data", outputs="image_data"),
            Rescale(inputs="image_data", outputs="image_data"),
            AddChannel(inputs="image_data", outputs="image_data", num_channel=1 if pretrain == "tencent" else 3),
            Delete(keys="partition"),
            Delete(keys="filename", mode="!test")
        ])
    if pretrain == "tencent":
        backbone = fe.build(
            model_fn=lambda: R50TencentEncoder(block=Bottleneck, layers=[3, 4, 6, 3], weight_path=weight_path),
            optimizer_fn=lambda x: torch.optim.Adam(x, lr=init_lr),
            model_name="backbone")
    else:
        backbone = fe.build(model_fn=lambda: R50TorchVideoEncoder(pretrain=pretrain, weight_path=weight_path),
                            optimizer_fn=lambda x: torch.optim.Adam(x, lr=init_lr),
                            model_name="backbone")
    classifier = fe.build(model_fn=Classifier,
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
        AUC(inputs=("pred", "label"), outputs="auc", mode=("eval", "test")),
        CSVLogger(filename=os.path.join(output_dir, "test_result.csv"), instance_id_key="filename", mode="test"),
        BestModelSaver(model=backbone, save_dir=output_dir, metric="auc", save_best_mode="max"),
        BestModelSaver(model=classifier, save_dir=output_dir, metric="auc", save_best_mode="max")
    ]
    est = fe.Estimator(pipeline=pipeline,
                       network=network,
                       epochs=epochs,
                       traces=traces,
                       log_steps=log_steps,
                       train_steps_per_epoch=train_steps_per_epoch)
    return est
