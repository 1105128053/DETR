import torch
# 下载地址： https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
pretrained_weights = torch.load('./detr-r50-e632da11.pth')

num_class = 3
pretrained_weights["model"]["class_embed.weight"].resize_(num_class+1,256)
pretrained_weights["model"]["class_embed.bias"].resize_(num_class+1)
torch.save(pretrained_weights,"detr-r50_%d.pth"%num_class)