from torchvision import transforms


class TransformCIFARTrainBaseline(object):

    def __init__(self,
        norm_mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        norm_std=[x / 255.0 for x in [63.0, 62.1, 66.7]]):
        super().__init__()

        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])


    def __call__(self, img):
        return self.transform(img)