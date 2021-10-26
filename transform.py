from torchvision import transforms


INPUT_SIZE = (384, 384)


def get_transform():
    return transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(INPUT_SIZE, (0.8, 1.), (1., 1.)),
        transforms.ToTensor(),
    ])


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from Dataset import *

    img_labels = ImageLabels('../datasets/training_labels.txt')
    dataset = MyDataset('../datasets/training_images', img_labels=img_labels, transform=get_transform())

    label, img = dataset[5]
    print(img.shape)
    print(label)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
