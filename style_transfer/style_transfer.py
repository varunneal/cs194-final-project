import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import PIL
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

SQUEEZENET_MEAN = [0.485, 0.456, 0.406]
SQUEEZENET_STD = [0.229, 0.224, 0.225]

device = th.device("mps")


def get_cnn():
    """
    squeezenet1_1 notes
    pretrained ofc

    ---
    inference transforms do the following:
    Accepts PIL.Image batched (B,C,H,W) and single (C,H,W) th.Tensor objects
    Images are resized to resize_size=[256] using some interpolation
    And are also cropped
    Values are first rescaled to [0.0, 1.0] then normalized using
    mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].

    - we would then need to reverse this process I suppose, idk
    """
    cnn = th.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1',
                      weights='SqueezeNet1_1_Weights.DEFAULT')  # 52 parameters
    for param in cnn.parameters():
        param.requires_grad = False
    return cnn


"""
good reading on transformations
https://pytorch.org/vision/stable/transforms.html
"""


def preprocess(img_path, size=512):
    img = Image.open(img_path).convert('RGB')
    transform = T.Compose([
        T.Resize(size),  # rescales smallest dim of image to match size
        T.ToTensor(),  # converts to pytorch tensor (good time to move to device?)
        T.Normalize(mean=SQUEEZENET_MEAN,
                    std=SQUEEZENET_STD),
        # mean is a sequence of means for each channel
        # std is a sequence of means for each channel
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img).to(device, th.float)


def deprocess(img):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD]),
        T.Normalize(mean=[-m for m in SQUEEZENET_MEAN], std=[1, 1, 1]),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])
    return transform(img.cpu())


def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def extract_features(model, image):
    """
    Takes in an image, a model (cnn) and returns a list of feature
    maps (one per layer).
    """
    features = []
    prev_feat = image

    for i, module in enumerate(model.features):
        # print(i, "->", module)
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features


def content_loss(weight, content_current, content_target):
    return weight * (content_current - content_target).square().sum()


def style_loss(feats, style_layers, style_targets, weights):
    style_loss = 0
    for i in range(len(style_layers)):
        loss_i = content_loss(weights[i],
                              gram_matrix(feats[style_layers[i]]), style_targets[i])
        style_loss += loss_i
    return style_loss


def variance(img, weight):
    first = (img[:, :, :, :-1] - img[:, :, :, 1:]).pow(2).sum()
    second = (img[:, :, :-1, :] - img[:, :, 1:, :]).pow(2).sum()
    var = weight * (first + second)
    return var


def gram_matrix(features, normalize=True):
    N, C, H, W = features.size()
    features = features.reshape(N, C, -1)
    gram = th.ones((N, C, C))
    for i in range(N):
        gram[i] = th.mm(features[i], features[i].t())
    if normalize:
        gram = gram / (H * W * C)
    return gram


def style_transfer(model, content_img_path, style_img_path, image_size, style_size, content_layer,
                   content_weight, style_layers, style_weights, var_weight):
    ### Extract features for content image ###
    content_img = preprocess(content_img_path, size=image_size)  # pytorch FloatTensor on device

    content_features = extract_features(model, content_img)  # , content_layer)
    content_target = content_features[content_layer].clone()

    ### Extract features for the style image ####
    style_img = preprocess(style_img_path, size=style_size)

    style_features = extract_features(model, style_img)  # , style_layers)
    style_targets = []
    for idx in style_layers:
        style_targets.append(gram_matrix(style_features[idx].clone()))

    ### Initialize output image to content image ###
    out_img = content_img.clone().to(device, th.float)

    ### Compute gradient on our img ###
    img_param = nn.Parameter(out_img).to(device)  # aka img_var

    lr = 3.0
    decay_lr = 0.1
    decay_at = 180

    optimizer = th.optim.Adam([img_param], lr=lr)

    for t in range(300):
        if t < 190:
            out_img.clamp_(-1.5, 1.5)
        optimizer.zero_grad()

        feats = extract_features(model, img_param)

        c_loss = content_loss(content_weight, feats[content_layer], content_target)
        s_loss = style_loss(feats, style_layers, style_targets, style_weights)
        v_loss = variance(img_param, var_weight)

        loss = c_loss + s_loss + v_loss

        loss.backward()  # this updates only img_param
        # since we turned off grad on all other params
        if t == decay_at:
            optimizer = th.optim.Adam([img_param], lr=decay_lr)
        optimizer.step()

        if t % 100 == 0:
            print("Iteration {}".format(t))
            # imshow deprocess(out_img.cpu())

    # finished
    # plt.imshow(deprocess(out_img.cpu()))
    # plt.show()
    return deprocess(out_img)


def default_style_transfer(model, img_path, style_path):
    params = {
        'model': model,
        'content_img_path': img_path,
        'style_img_path': style_path,
        'image_size': 192 * 3,
        'style_size': 256 * 5,
        'content_layer': 3,
        'content_weight': 1e-2,
        'style_layers': (1, 4, 6, 7),
        'style_weights': (200000, 5000, 120, 10),
        'var_weight': 5e-2
    }

    return style_transfer(**params)


def bulk_style(cnn, source):
    content_path = f'lib/{source}.jpg'

    for style in ['thewave', 'starrynight']:

        style_path = f'lib/{style}.jpg'
        out_path = f'out/{source}_{style}.jpg'
        try:
            out_img = default_style_transfer(cnn, content_path, style_path)
            out_img.save(out_path)
            # Image.sav(out_path, out_img)
        except:
            print("failed for", source, ",", style)
            pass


def main():
    cnn = get_cnn().to(device)
    params = {
        'model': cnn,
        'content_img_path': "lib/post.jpg",
        'style_img_path': "lib/alexgrey.jpg",
        'image_size': 192 * 3,
        'style_size': 256 * 5,
        'content_layer': 3,
        'content_weight': 5e-2,
        'style_layers': (1, 4, 6, 7),
        'style_weights': (200000, 5000, 120, 10),
        'var_weight': 1e-1
    }
    s = style_transfer(**params)
    plt.imshow(s)
    s.save("test.jpg")
    # plt.imsave("test.jpg", s)
    plt.show()

    # bulk_style(cnn, 'post')
    # for source in ['sjerome', 'Oxford', 'doe', 'campanile', 'bay']:
    #     bulk_style(cnn, source)


if __name__ == '__main__':
    main()
