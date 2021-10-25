import torch
import torchvision
import Dataset as D
import transform as T
from torch.utils.data import DataLoader
from validation import validation


lr = 0.05
batch_size = 32

# dataset
img_labels = D.ImageLabels('../datasets/training_labels.txt')
dataset = D.MyDataset('../datasets/training_images', img_labels=img_labels, transform=T.get_transform(), val=0.15)
class_transformer = D.ClassTransform('../datasets/classes.txt')
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# model
model = torch.nn.Sequential(
    torchvision.models.resnet50(pretrained=True),
    torch.nn.Linear(1000, class_transformer.total_size),
)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# loss
loss_function = torch.nn.CrossEntropyLoss()

# training loop
epoch = 100
for i in range(epoch):
    # train
    model.train()
    for labels, images in data_loader:
        y_true = torch.tensor(class_transformer.to_order(labels))
        y_pred = model(images)

        loss = loss_function(y_pred, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
    # validation
    model.eval()
    validation(model, dataset)
