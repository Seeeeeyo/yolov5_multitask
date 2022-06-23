import pytest
from models.yolo import Model
import torch

########################################
shapes_base = {
    0: [4, 32, 320, 320],
    1: [4, 64, 160, 160],
    2: [4, 64, 160, 160],
    3: [4, 128, 80, 80],
    4: [4, 128, 80, 80],
    5: [4, 256, 40, 40],
    6: [4, 256, 40, 40],
    7: [4, 512, 20, 20],
    8: [4, 512, 20, 20],
    9: [4, 512, 20, 20],
    10: [4, 256, 20, 20],
    11: [4, 256, 40, 40],
    12: [4, 512, 40, 40],
    13: [4, 256, 40, 40],
    14: [4, 128, 40, 40],
    15: [4, 128, 80, 80],
    16: [4, 256, 80, 80],
    17: [4, 128, 80, 80],
    18: [4, 128, 40, 40],
    19: [4, 256, 40, 40],
    20: [4, 256, 40, 40],
    21: [4, 256, 20, 20],
    22: [4, 512, 20, 20],
    23: [4, 512, 20, 20]
    # No layer added
}
cfg_base = "yolov5s_base.yaml"
########################################
shapes_avgpool_all_fm = {
    0: [4, 32, 320, 320],
    1: [4, 64, 160, 160],
    2: [4, 64, 160, 160],
    3: [4, 128, 80, 80],
    4: [4, 128, 80, 80],
    5: [4, 256, 40, 40],
    6: [4, 256, 40, 40],
    7: [4, 512, 20, 20],
    8: [4, 512, 20, 20],
    9: [4, 512, 20, 20],
    10: [4, 256, 20, 20],
    11: [4, 256, 40, 40],
    12: [4, 512, 40, 40],
    13: [4, 256, 40, 40],
    14: [4, 128, 40, 40],
    15: [4, 128, 80, 80],
    16: [4, 256, 80, 80],
    17: [4, 128, 80, 80],
    18: [4, 128, 40, 40],
    19: [4, 256, 40, 40],
    20: [4, 256, 40, 40],
    21: [4, 256, 20, 20],
    22: [4, 512, 20, 20],
    23: [4, 512, 20, 20],
    # added layers
    24: [4, 256, 1, 1],
    25: [4, 128, 1, 1],
    26: [4, 128, 1, 1],
    27: [4, 512, 1, 1],
    # detection head
    28: [4, 3],
}
cfg_avgpool_all_fm = "yolov5s_avgpool_all_fm.yaml"
##########################################
shapes_complex = {
    0: [4, 32, 320, 320],
    1: [4, 64, 160, 160],
    2: [4, 64, 160, 160],
    3: [4, 128, 80, 80],
    4: [4, 128, 80, 80],
    5: [4, 256, 40, 40],
    6: [4, 256, 40, 40],
    7: [4, 512, 20, 20],
    8: [4, 512, 20, 20],
    9: [4, 512, 20, 20],
    10: [4, 256, 20, 20],
    11: [4, 256, 40, 40],
    12: [4, 512, 40, 40],
    13: [4, 256, 40, 40],
    14: [4, 128, 40, 40],
    15: [4, 128, 80, 80],
    16: [4, 256, 80, 80],
    17: [4, 128, 80, 80],
    18: [4, 128, 40, 40],
    19: [4, 256, 40, 40],
    20: [4, 256, 40, 40],
    21: [4, 256, 20, 20],
    22: [4, 512, 20, 20],
    23: [4, 512, 20, 20],
    # added layers
    24: [4, 128, 20, 20],
    25: [4, 128, 40, 40],
    26: [4, 128, 40, 40],
    27: [4, 384, 40, 40],
    # detection head
    28: [4, 3],
}
cfg_complex = "yolov5s_complex.yaml"
########################################
shapes_avgpool_fm_large = {
    0: [4, 32, 320, 320],
    1: [4, 64, 160, 160],
    2: [4, 64, 160, 160],
    3: [4, 128, 80, 80],
    4: [4, 128, 80, 80],
    5: [4, 256, 40, 40],
    6: [4, 256, 40, 40],
    7: [4, 512, 20, 20],
    8: [4, 512, 20, 20],
    9: [4, 512, 20, 20],
    10: [4, 256, 20, 20],
    11: [4, 256, 40, 40],
    12: [4, 512, 40, 40],
    13: [4, 256, 40, 40],
    14: [4, 128, 40, 40],
    15: [4, 128, 80, 80],
    16: [4, 256, 80, 80],
    17: [4, 128, 80, 80],
    18: [4, 128, 40, 40],
    19: [4, 256, 40, 40],
    20: [4, 256, 40, 40],
    21: [4, 256, 20, 20],
    22: [4, 512, 20, 20],
    23: [4, 512, 20, 20],
    # added layers
    24: [4, 256, 1, 1],
    # detection head
    25: [4, 3],
}
cfg_avgpool_fm_large = "yolov5s_avgpool_fm_large.yaml"


########################################


def validate_yolo5s(shapes=shapes_base, cfg="yolov5s_base.yaml"):
    cfg = cfg
    device = "cuda"
    batch = 4
    shapes_detect = {
        0: [4, 3, 80, 80, 85],
        1: [4, 3, 40, 40, 85],
        2: [4, 3, 20, 20, 85],
    }

    x = torch.rand(batch, 3, 640, 640).to(device)
    model = Model(cfg).to(device)
    y = []  # outputs
    # print('Config: ', cfg)
    # print("Added layers:")
    for m in model.model:
        if m.f != -1:  # if not from previous layer
            x = (
                y[m.f]
                if isinstance(m.f, int)
                else [x if j == -1 else y[j] for j in m.f]
            )  # from earlier layers
        x = m(x)  # run
        if m._get_name() != "Detect":
            assert list(x.shape) == shapes[m.i]
        else:
            assert m.i == max(shapes.keys()) + 1
            assert m._get_name() == "Detect"
            assert len(x) == 3
            for i in range(3):
                assert list(x[i].shape) == shapes_detect[i]
                assert (torch.sum(x[i]).item()) != 0

        # if m.i > 23 and m._get_name() != 'Detect':
        # print("layer", m.i, '---', m._get_name(), "--- Output shape:", x.shape)
        y.append(x)  # save output
    return model


def test_yolo5s_base():
    # this is simply the basic Yolov5 small model
    model = validate_yolo5s(shapes=shapes_base, cfg=cfg_base)
    assert model.model[24]._get_name() == "Detect"


def test_yolo5s_avgpool_all_fm():
    # 1) applied adaptive average pooling to the feature maps large, medium and small (20, 40, 80).
    # 2) Then concatenated these 1x1x256 -- 1x1x128 -- 1x1x128 together to result into 1x1x512.
    # 3) Then squeezed this 1x1x512 and finally applied a Linear layer for classification
    model = validate_yolo5s(shapes=shapes_avgpool_all_fm, cfg=cfg_avgpool_all_fm)
    assert model.model[24]._get_name() == "AdaptiveAvgPool2d"
    assert model.model[25]._get_name() == "AdaptiveAvgPool2d"
    assert model.model[26]._get_name() == "AdaptiveAvgPool2d"
    assert model.model[27]._get_name() == "Concat"
    assert model.model[28]._get_name() == "LinearSqueeze"


def test_yolo5s_complex():
    # 1) Applied a convolution to the feature map-large: from 20x20x256 to 20x20x128
    # 2) then applied an upsampling on this 20x20x128 to get a 40x40x128
    # No change to the feature map-medium (already 40x40x256)
    # 3) Applied an avg pooling to the feature map-small: from 80x80x128 to 40x40x128
    # 4) Concatenated the upsample feature map-large with the fm-m and avg pooled fm-s resulting in: 40x40x384
    # 5) Then applied Yolo's classification head which is:
    # 5.1) adaptive average pooling to get 1x1x384 (+ cat if it's a list)
    # 5.2) 2D convolution resulting in 1x1x num class
    # 5.3) flatten the result
    model = validate_yolo5s(shapes=shapes_complex, cfg=cfg_complex)
    # fm-large
    assert model.model[24]._get_name() == "Conv"
    assert model.model[25]._get_name() == "Upsample"
    # fm-small
    assert model.model[26]._get_name() == "AvgPool2d"
    # concat all fm
    assert model.model[27]._get_name() == "Concat"

    assert model.model[28]._get_name() == "Classify"


def test_yolov5s_avgpool_fm_large():
    # applied Adaptive average pooling on the fm-large (20x20x256) to result into 1x1x256
    # Squeezed the result and applied a Linear layer
    model = validate_yolo5s(shapes=shapes_avgpool_fm_large, cfg=cfg_avgpool_fm_large)
    # fm-large
    assert model.model[24]._get_name() == "AdaptiveAvgPool2d"

    assert model.model[25]._get_name() == "LinearSqueeze"
