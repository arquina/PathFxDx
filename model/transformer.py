import torch
import timm

class CustomViT(torch.nn.Module):
    def __init__(self, num_features = 768, pretrained=True):
        super(CustomViT, self).__init__()

        # Load the pre-trained ViT model
        self.original_vit = timm.create_model('vit_base_patch16_224_dino', pretrained=pretrained, num_classes=0)
        self.fracture_vit = timm.create_model('vit_base_patch16_224_dino', pretrained= pretrained, num_classes = 0)
        self.binary_head = torch.nn.Linear(num_features * 2, 2)
        self.num_features = num_features

    def forward(self, original, fracture):
        original = self.original_vit.forward_features(original)[:, 0, :]
        fracture = self.fracture_vit.forward_features(fracture)[:, 0, :]
        original = original.reshape((-1, self.num_features))
        fracture = fracture.reshape((-1, self.num_features))
        x = torch.concat((original, fracture), axis = 1)
        x = x.reshape((-1, self.num_features * 2))
        binary_out = self.binary_head(x)

        return binary_out
