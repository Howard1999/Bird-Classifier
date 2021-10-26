import torch
import torchvision
from torchvision import transforms
from transform import INPUT_SIZE
from Dataset import ClassTransform
from PIL import Image
from pytorch_pretrained_vit import ViT


# class transform
classes_path = '../datasets/classes.txt'
class_transform = ClassTransform(classes_path)

# load model
model_path = '../record/v1/baseline'
model = torch.nn.Sequential(
    ViT('B_16_imagenet1k', pretrained=True),
    torch.nn.Linear(1000, class_transform.total_size)
)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(model_path, map_location="cuda:0"))
model.to('cuda:0')
model.eval()

# image transform
transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
])

# read test order file
test_order_path = '../datasets/testing_img_order.txt'
with open(test_order_path) as fp:
    test_list = fp.read().splitlines()

# generate prediction file
output_path = './predict.txt'
with open(output_path, 'w') as fp:
    test_folder = '../datasets/testing_images/'
    for img_name in test_list:
        image = transform(Image.open(test_folder+img_name))

        y_pred = model(image[None, :].to('cuda:0'))
        class_name = class_transform.to_class_name(y_pred[0])

        fp.write(img_name+' '+class_name+'\n')
        print(img_name+' '+class_name)
