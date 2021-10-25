import torch
import torchvision
import Dataset as D
import transform as T
from torch.utils.data import DataLoader
from validation import validation
from plot import show_hist


lr = 0.001
batch_size = 32

# dataset
img_labels = D.ImageLabels('../datasets/training_labels.txt')
dataset, val_dataset = D.get_train_val_dataset('../datasets/training_images', img_labels=img_labels,
                                               transform=T.get_transform(), val=0.15)
class_transformer = D.ClassTransform('../datasets/classes.txt')
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# model
model = torch.nn.Sequential(
    torchvision.models.resnet50(pretrained=True),
    torch.nn.Linear(1000, class_transformer.total_size),
)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

# loss
loss_function = torch.nn.CrossEntropyLoss()

# training history
train_loss, validation_loss = [], []
train_acc, validation_acc = [], []

# training loop
epoch = 100
for i in range(epoch):
    # train
    model.train()
    loss_sum, positive, cnt = 0., 0, 0
    for labels, images in data_loader:
        y_true = torch.tensor(class_transformer.to_order(labels))
        y_pred = model(images)

        loss = loss_function(y_pred, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # metrics
        cnt += 1
        loss_sum += loss.item()
        pre_class = class_transformer.to_class_name(y_pred)
        for ind, pre_label in enumerate(pre_class):
            if pre_class == labels[ind]:
                positive += 1

    # validation
    model.eval()
    metrics = validation(model, val_dataset, class_transformer, loss_function)

    # record the loss
    train_loss.append(loss_sum/cnt)
    validation_loss.append(metrics['loss'])

    # record the acc
    train_loss.append(positive / len(dataset))
    validation_loss.append(metrics['acc'])

    print('epoch', i, 'finish')
    print(metrics)

show_hist([train_loss, validation_loss], 'loss', ['train', 'validation'])
show_hist([train_acc, validation_acc], 'acc', ['train', 'validation'])
