import os
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def _read_image(img_path):
    return Image.open(img_path, mode='r').convert('RGB')


_TRAINING = 0
_VALIDATION = 1


class MyDataset(Dataset):

    def __init__(self, img_dir, transform=transforms.ToTensor(), img_labels=None, mode=_TRAINING, val=0.):
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        self.img_labels = img_labels
        self.transform = transform
        self.mode = mode

        # check all img has label
        if self.img_labels is not None:
            for img_name in self.img_list:
                if self.img_labels[img_name] is None:
                    raise Exception('[MyDataset]img_labels is given, but not every image has label: '+img_name)

        # check val
        if not 0. <= val <= 1.:
            raise Exception('[MyDataset]val should between 0. ~ 1.')

        # mode handle
        if self.mode == _TRAINING:
            self._offset = 0
            self.size = int(len(self.img_list)*(1-val))
        elif self.mode == _VALIDATION:
            self._offset = int(len(self.img_list) * (1 - val))
            self.size = len(self.img_list) - self._offset
        else:
            raise Exception('[MyDataset]mode should be 0.TRAINING or VALIDATION')

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        index += self._offset

        _label = self.img_labels[self.img_list[index]] if self.img_labels is not None else None
        _img = _read_image(self.img_dir+'/'+self.img_list[index])
        _img = self.transform(_img)

        return _label, _img


class ImageLabels:

    def __init__(self, label_path):
        self.labels = dict()
        with open(label_path) as fp:
            for img_label in fp.readlines():
                img_label = img_label.split()
                self.labels[img_label[0]] = img_label[1]

    def __getitem__(self, img_name):
        if img_name in self.labels:
            return self.labels[img_name]
        return None


class ClassTransform: # convert between one hot tensor and class name

    def __init__(self, class_path):
        self.cls2order = dict()
        self.order2cls = dict()
        self.total_size = 0
        with open(class_path) as fp:
            for ind, class_name in enumerate(fp.readlines()):
                class_name = class_name.replace('\n', '')
                if class_name != '':
                    self.cls2order[class_name] = ind
                    self.order2cls[ind] = class_name
                    self.total_size += 1

    def to_one_hot(self, cls):
        if type(cls) == list or type(cls) == tuple:
            one_hot = torch.zeros(len(cls), self.total_size)
            for idx, class_name in enumerate(cls):
                one_hot[idx][self.cls2order[class_name]] = 1.
        else:
            one_hot = torch.zeros(self.total_size)
            one_hot[self.cls2order[cls]] = 1.
        return one_hot

    def to_order(self, cls):
        if type(cls) == list or type(cls) == tuple:
            order = []
            for class_name in cls:
                order.append(self.cls2order[class_name])
        else:
            order = self.cls2order[cls]
        return order

    def to_class_name(self, one_hot: torch.Tensor):
        if type(one_hot) != torch.Tensor:
            raise Exception('[ClassTransform]one_hot should be torch.Tensor type')
        shape = one_hot.shape
        if len(shape) == 2 and shape[1] == self.total_size:
            class_name = []
            _, order = torch.topk(one_hot, 1)
            for ind in order:
                class_name.append(self.order2cls[ind[0].item()])
        elif len(shape) == 1 and shape[0] == self.total_size:
            _, order = torch.topk(one_hot, 1)
            class_name = self.order2cls[order[0].item()]
        else:
            raise Exception('[ClassTransform]Tensor Shape Error: '+shape)
        return class_name


def get_train_val_dataset(*args, **kwargs):
    train_dataset = MyDataset(*args, **kwargs, mode=_TRAINING)
    val_dataset = MyDataset(*args, **kwargs, mode=_VALIDATION)
    return train_dataset, val_dataset


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    img_labels = ImageLabels('../datasets/training_labels.txt')
    dataset = MyDataset('../datasets/training_images', img_labels=img_labels)

    # test read data
    label, img = dataset[5]

    print(label)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

    # test convert label
    t = ClassTransform('../datasets/classes.txt')

    print(label)
    one_hot = t.to_one_hot(label)
    print(one_hot.shape)
    print(t.to_class_name(one_hot))

    labels = []
    for i in range(5):
        label, _ = dataset[i]
        labels.append(label)
    print(labels)
    one_hot = t.to_one_hot(labels)
    print(one_hot.shape)
    print(t.to_class_name(one_hot))
