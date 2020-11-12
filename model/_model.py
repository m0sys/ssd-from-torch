import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

from utils.util import decimate


class VGGBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(3, 1, padding=1)

        # Replacement for FC6 and FC7 in VGG16.
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.load_pretrained_layers()

    def load_pretrained_layers(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        self._transfer_pretrained_weights(
            state_dict, param_names, pretrained_state_dict, pretrained_param_names
        )

        conv_fc6_weight = self._convert_fc_to_conv(
            pretrained_state_dict["classifier.0.weight"], (4096, 512, 7, 7)
        )
        conv_fc6_bias = pretrained_state_dict["classifier.0.bias"]
        state_dict["conv6.weight"] = self._subsample_to_conv_sz(
            conv_fc6_weight, m=[4, None, 3, 3]
        )
        state_dict["conv6.bias"] = self._subsample_to_conv_sz(conv_fc6_bias, m=[4])
        assert conv_fc6_weight.shape == (4096, 512, 7, 7)
        assert conv_fc6_bias.shape == (4096,)
        assert state_dict["conv6.weight"].shape == (1024, 512, 3, 3)
        assert state_dict["conv6.bias"].shape == (1024,)

        conv_fc7_weight = self._convert_fc_to_conv(
            pretrained_state_dict["classifier.3.weight"], (4096, 4096, 1, 1)
        )
        conv_fc7_bias = pretrained_state_dict["classifier.3.bias"]
        state_dict["conv7.weight"] = self._subsample_to_conv_sz(
            conv_fc7_weight, m=[4, 4, None, None]
        )
        state_dict["conv7.bias"] = self._subsample_to_conv_sz(conv_fc7_bias, m=[4])
        assert conv_fc7_weight.shape == (4096, 4096, 1, 1)
        assert conv_fc6_bias.shape == (4096,)
        assert state_dict["conv7.weight"].shape == (1024, 1024, 1, 1)
        assert state_dict["conv7.bias"].shape == (1024,)

        self.load_state_dict(state_dict)
        print("\nLoaded base model.\n")

    def _transfer_pretrained_weights(
        self,
        model_state_dict,
        model_param_names,
        pretrained_state_dict,
        pretrained_param_names,
    ):
        for i, param in enumerate(model_param_names[:-4]):
            model_state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

    def _convert_fc_to_conv(self, fc_weights, conv_dims):
        return fc_weights.view(*conv_dims)

    def _subsample_to_conv_sz(self, conv_weight, m):
        return decimate(conv_weight, m=m)

    def forward(self, x):
        out = F.relu(self.conv1_1(x))
        out = F.relu(self.conv1_2(out))
        out = self.pool1(out)

        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = self.pool2(out)

        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out))
        out = F.relu(self.conv3_3(out))
        out = self.pool3(out)

        out = F.relu(self.conv4_1(out))
        out = F.relu(self.conv4_2(out))
        out = F.relu(self.conv4_3(out))
        conv4_3_feats = out
        out = self.pool4(out)

        out = F.relu(self.conv5_1(out))
        out = F.relu(self.conv5_2(out))
        out = F.relu(self.conv5_3(out))
        out = self.pool5(out)

        out = F.relu(self.conv6(out))
        conv7_feats = F.relu(self.conv7(out))

        # Lower-level feature maps.
        return conv4_3_feats, conv7_feats


class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self):
        super().__init__()

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """

        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.0)

    def forward(self, conv7_feats):
        bs = conv7_feats.size(0)

        out = F.relu(self.conv8_1(conv7_feats))
        out = F.relu(self.conv8_2(out))
        conv8_2_feats = out
        assert conv8_2_feats.shape == (bs, 512, 10, 10)

        out = F.relu(self.conv9_1(out))
        out = F.relu(self.conv9_2(out))
        conv9_2_feats = out
        assert conv9_2_feats.shape == (bs, 256, 5, 5)

        out = F.relu(self.conv10_1(out))
        out = F.relu(self.conv10_2(out))
        conv10_2_feats = out
        assert conv10_2_feats.shape == (bs, 256, 3, 3)

        out = F.relu(self.conv11_1(out))
        conv11_2_feats = F.relu(self.conv11_2(out))
        assert conv11_2_feats.shape == (bs, 256, 1, 1)

        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats


class PredictionConvolutions(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and
    higher-level feature maps.
    """

    def __init__(self, n_classes):
        super().__init__()

        self.n_classes = n_classes

        # k denotes the number of default boxes per feature cell in paper.
        n_boxes = {
            "conv4_3": 4,  # k = 4
            "conv7": 6,  # k = 6
            "conv8_2": 6,  # k = 6
            "conv9_2": 6,  # k = 6
            "conv10_2": 4,  # k = 4
            "conv11_2": 4,  # k = 4
        }

        # Localization prediction convs (predict offsets w.r.t. prior-boxes).
        self.loc_conv4_3 = nn.Conv2d(
            512, n_boxes["conv4_3"] * 4, kernel_size=3, padding=1
        )
        self.loc_conv7 = nn.Conv2d(1024, n_boxes["conv7"] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(
            512, n_boxes["conv8_2"] * 4, kernel_size=3, padding=1
        )
        self.loc_conv9_2 = nn.Conv2d(
            256, n_boxes["conv9_2"] * 4, kernel_size=3, padding=1
        )
        self.loc_conv10_2 = nn.Conv2d(
            256, n_boxes["conv10_2"] * 4, kernel_size=3, padding=1
        )
        self.loc_conv11_2 = nn.Conv2d(
            256, n_boxes["conv11_2"] * 4, kernel_size=3, padding=1
        )

        # Class prediction convolutions (predict classes in localization boxes).
        self.cl_conv4_3 = nn.Conv2d(
            512, n_boxes["conv4_3"] * n_classes, kernel_size=3, padding=1
        )
        self.cl_conv7 = nn.Conv2d(
            1024, n_boxes["conv7"] * n_classes, kernel_size=3, padding=1
        )
        self.cl_conv8_2 = nn.Conv2d(
            512, n_boxes["conv8_2"] * n_classes, kernel_size=3, padding=1
        )
        self.cl_conv9_2 = nn.Conv2d(
            256, n_boxes["conv9_2"] * n_classes, kernel_size=3, padding=1
        )
        self.cl_conv10_2 = nn.Conv2d(
            256, n_boxes["conv10_2"] * n_classes, kernel_size=3, padding=1
        )
        self.cl_conv11_2 = nn.Conv2d(
            256, n_boxes["conv11_2"] * n_classes, kernel_size=3, padding=1
        )

        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """

        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.0)

    def forward(
        self,
        conv4_3_feats,
        conv7_feats,
        conv8_2_feats,
        conv9_2_feats,
        conv10_2_feats,
        conv11_2_feats,
    ):
        bs = conv4_3_feats.size(0)

        # Localization predictions.
        l_conv4_3 = self._apply_loc_layer(conv4_3_feats, "loc_conv4_3", 16, 38, 38)

        l_conv7 = self._apply_loc_layer(conv7_feats, "loc_conv7", 24, 19, 19)

        l_conv8_2 = self._apply_loc_layer(conv8_2_feats, "loc_conv8_2", 24, 10, 10)

        l_conv9_2 = self._apply_loc_layer(conv9_2_feats, "loc_conv9_2", 24, 5, 5)

        l_conv10_2 = self._apply_loc_layer(conv10_2_feats, "loc_conv10_2", 16, 3, 3)

        l_conv11_2 = self._apply_loc_layer(conv11_2_feats, "loc_conv11_2", 16, 1, 1)

        # Classification prediction within localizations.
        c_conv4_3 = self._apply_cl_layer(conv4_3_feats, "cl_conv4_3", 38, 38, 4)

        c_conv7 = self._apply_cl_layer(conv7_feats, "cl_conv7", 19, 19, 6)

        c_conv8_2 = self._apply_cl_layer(conv8_2_feats, "cl_conv8_2", 10, 10, 6)

        c_conv9_2 = self._apply_cl_layer(conv9_2_feats, "cl_conv9_2", 5, 5, 6)

        c_conv10_2 = self._apply_cl_layer(conv10_2_feats, "cl_conv10_2", 3, 3, 4)

        c_conv11_2 = self._apply_cl_layer(conv11_2_feats, "cl_conv11_2", 1, 1, 4)

        locs = torch.cat(
            [l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1
        )
        assert locs.shape == (bs, 8732, 4)

        class_scores = torch.cat(
            [c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2],
            dim=1,
        )
        assert class_scores.shape == (bs, 8732, self.n_classes)

        return locs, class_scores

    def _apply_loc_layer(self, x, layer_name, n_points, h, w):
        bs = x.size(0)
        total_boxes = int(h * w * n_points / 4)

        out = getattr(self, layer_name)(x)
        assert out.shape == (bs, n_points, h, w)

        out = out.permute(0, 2, 3, 1).contiguous()
        assert out.shape == (bs, h, w, n_points)

        out = out.view(bs, -1, 4)
        assert out.shape == (bs, total_boxes, 4)

        return out

    def _apply_cl_layer(self, x, layer_name, h, w, num_boxes):
        bs = x.size(0)
        total_boxes = int(h * w * num_boxes)

        out = getattr(self, layer_name)(x)
        assert out.shape == (bs, num_boxes * self.n_classes, h, w)

        out = out.permute(0, 2, 3, 1).contiguous()
        assert out.shape == (bs, w, h, num_boxes * self.n_classes)

        out = out.view(bs, -1, self.n_classes)
        assert out.shape == (bs, total_boxes, self.n_classes)

        return out
