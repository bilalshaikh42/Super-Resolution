import torch
import utility
import model
import copy

# if checkpoint.ok:
#     _model = model.Model(args, checkpoint)
#     print('imported')

model = torch.load('/home/ctebright/EDSR-PyTorch/pretrained_models/EDSR_x4.pt')
weights = [i for i in model if 'weight' in i]

# print('all layers')
# for i in model:
#     print(i)

# print()
# print()
# print()
# print()

# Loop to zero each block individually
# print('weights testing')
for layer in weights:
    # print(layer)
    if 'body.0.w' in layer:
        # print('resblock layer')
        # print(layer)
        base = layer[0:-9]
        weight0 = base + '.0.weight'
        bias0 = base + '.0.bias'
        weight2 = base + '.2.weight'
        bias2 = base + '.2.bias'
        # print(base)
        # print(weight0, bias0, weight2, bias2)
        model_temp = copy.copy(model)
        model_temp[weight0] = torch.zeros(model_temp[weight0].shape)
        model_temp[bias0] = torch.zeros(model_temp[bias0].shape)
        model_temp[weight2] = torch.zeros(model_temp[weight2].shape)
        model_temp[bias2] = torch.zeros(model_temp[bias2].shape)
        torch.save(model_temp,'/home/ctebright/EDSR-PyTorch/zeroed/{}.pt'.format(layer[0:-14]))
    else:
        # print('not resblock layer')
        # print(layer)
        bias = layer[0:-6]+'bias'
        # print(layer, bias)
        model_temp = copy.copy(model)
        model_temp[layer] = torch.zeros(model_temp[layer].shape)
        model_temp[bias] = torch.zeros(model_temp[bias].shape)
        torch.save(model_temp,'/home/ctebright/EDSR-PyTorch/zeroed/{}.pt'.format(layer[0:-7]))


# Zero second conv of all blocks
model_temp = copy.copy(model)
for layer in weights:
    if 'body.2.w' in layer:
        bias = layer[0:-6]+'bias'
        # print(layer, bias)
        model_temp[layer] = torch.zeros(model_temp[layer].shape)
        model_temp[bias] = torch.zeros(model_temp[bias].shape)

torch.save(model_temp,'/home/ctebright/EDSR-PyTorch/zeroed/all_second_convs.pt')

# Zero first conv of all blocks
model_temp = copy.copy(model)
for layer in weights:
    if 'body.1.w' in layer:
        bias = layer[0:-6]+'bias'
        # print(layer, bias)
        model_temp[layer] = torch.zeros(model_temp[layer].shape)
        model_temp[bias] = torch.zeros(model_temp[bias].shape)

torch.save(model_temp,'/home/ctebright/EDSR-PyTorch/zeroed/all_first_convs.pt')

# Zero 4 blocks in middle
model_temp = copy.copy(model)
to_zero = ['body.14.body', 'body.15.body', 'body.16.body', 'body.17.body']
for layer in weights:
    for val in to_zero:
        if val in layer:
            bias = layer[0:-6]+'bias'
            # print(layer, bias)
            model_temp[layer] = torch.zeros(model_temp[layer].shape)
            model_temp[bias] = torch.zeros(model_temp[bias].shape)

torch.save(model_temp,'/home/ctebright/EDSR-PyTorch/zeroed/middle_4.pt')

# Zero first 4 blocks
model_temp = copy.copy(model)
to_zero = ['body.0.body', 'body.1.body', 'body.2.body', 'body.3.body']
for layer in weights:
    for val in to_zero:
        if val in layer:
            bias = layer[0:-6]+'bias'
            # print(layer, bias)
            model_temp[layer] = torch.zeros(model_temp[layer].shape)
            model_temp[bias] = torch.zeros(model_temp[bias].shape)

torch.save(model_temp,'/home/ctebright/EDSR-PyTorch/zeroed/first_4.pt')

# Zero last 4 blocks
model_temp = copy.copy(model)
to_zero = ['body.28.body', 'body.29.body', 'body.30.body', 'body.31.body']
for layer in weights:
    for val in to_zero:
        if val in layer:
            bias = layer[0:-6]+'bias'
            # print(layer, bias)
            model_temp[layer] = torch.zeros(model_temp[layer].shape)
            model_temp[bias] = torch.zeros(model_temp[bias].shape)

torch.save(model_temp,'/home/ctebright/EDSR-PyTorch/zeroed/last_4.pt')


# Zero 8 blocks in middle
model_temp = copy.copy(model)
to_zero = ['body.12.body', 'body.13.body', 'body.14.body', 'body.15.body', 'body.16.body', 'body.17.body', 'body.18.body', 'body.19.body']
for layer in weights:
    for val in to_zero:
        if val in layer:
            bias = layer[0:-6]+'bias'
            # print(layer, bias)
            model_temp[layer] = torch.zeros(model_temp[layer].shape)
            model_temp[bias] = torch.zeros(model_temp[bias].shape)

torch.save(model_temp,'/home/ctebright/EDSR-PyTorch/zeroed/middle_8.pt')

# Zero first 8 blocks
model_temp = copy.copy(model)
to_zero = ['body.0.body', 'body.1.body', 'body.2.body', 'body.3.body', 'body.4.body', 'body.5.body', 'body.6.body', 'body.7.body']
for layer in weights:
    for val in to_zero:
        if val in layer:
            bias = layer[0:-6]+'bias'
            # print(layer, bias)
            model_temp[layer] = torch.zeros(model_temp[layer].shape)
            model_temp[bias] = torch.zeros(model_temp[bias].shape)

torch.save(model_temp,'/home/ctebright/EDSR-PyTorch/zeroed/first_8.pt')

# Zero last 8 blocks
model_temp = copy.copy(model)
to_zero = ['body.24.body', 'body.25.body', 'body.26.body', 'body.27.body', 'body.28.body', 'body.29.body', 'body.30.body', 'body.31.body']
for layer in weights:
    for val in to_zero:
        if val in layer:
            bias = layer[0:-6]+'bias'
            # print(layer, bias)
            model_temp[layer] = torch.zeros(model_temp[layer].shape)
            model_temp[bias] = torch.zeros(model_temp[bias].shape)

torch.save(model_temp,'/home/ctebright/EDSR-PyTorch/zeroed/last_8.pt')


# Zero 16 blocks in middle
model_temp = copy.copy(model)
to_zero = ['body.8.body', 'body.9.body', 'body.10.body', 'body.11.body', 'body.12.body', 'body.13.body', 'body.14.body', 'body.15.body', 'body.16.body', 'body.17.body', 'body.18.body', 'body.19.body', 'body.20.body', 'body.21.body', 'body.22.body', 'body.23.body']
for layer in weights:
    for val in to_zero:
        if val in layer:
            bias = layer[0:-6]+'bias'
            # print(layer, bias)
            model_temp[layer] = torch.zeros(model_temp[layer].shape)
            model_temp[bias] = torch.zeros(model_temp[bias].shape)

torch.save(model_temp,'/home/ctebright/EDSR-PyTorch/zeroed/middle_16.pt')

# Zero first 16 blocks
model_temp = copy.copy(model)
to_zero = ['body.0.body', 'body.1.body', 'body.2.body', 'body.3.body', 'body.4.body', 'body.5.body', 'body.6.body', 'body.7.body', 'body.8.body', 'body.9.body', 'body.10.body', 'body.11.body', 'body.12.body', 'body.13.body', 'body.14.body', 'body.15.body']
for layer in weights:
    for val in to_zero:
        if val in layer:
            bias = layer[0:-6]+'bias'
            # print(layer, bias)
            model_temp[layer] = torch.zeros(model_temp[layer].shape)
            model_temp[bias] = torch.zeros(model_temp[bias].shape)

torch.save(model_temp,'/home/ctebright/EDSR-PyTorch/zeroed/first_16.pt')

# Zero last 16 blocks
model_temp = copy.copy(model)
to_zero = ['body.16.body', 'body.17.body', 'body.18.body', 'body.19.body', 'body.20.body', 'body.21.body', 'body.22.body', 'body.23.body', 'body.24.body', 'body.25.body', 'body.26.body', 'body.27.body', 'body.28.body', 'body.29.body', 'body.30.body', 'body.31.body']
for layer in weights:
    for val in to_zero:
        if val in layer:
            bias = layer[0:-6]+'bias'
            # print(layer, bias)
            model_temp[layer] = torch.zeros(model_temp[layer].shape)
            model_temp[bias] = torch.zeros(model_temp[bias].shape)

torch.save(model_temp,'/home/ctebright/EDSR-PyTorch/zeroed/last_16.pt')

model_temp = copy.copy(model)
for layer in weights:
    model_temp[layer] = torch.zeros(model_temp[layer].shape)
torch.save(model_temp,'/home/ctebright/EDSR-PyTorch/zeroed/all_zeros.pt')
