import Dataset as D
import torch


def validation(model, dataset: D.MyDataset, class_transformer: D.ClassTransform, loss_function, device):
    with torch.no_grad():
        positive = 0
        y_true, y_pred = [], []
        for label, image in dataset:
            y_true.append(class_transformer.to_order(label))
            y_pred.append(model(image[None, :]))

            pre_label = class_transformer.to_class_name(y_pred[-1][0])
            if pre_label == label:
                positive += 1
        loss = loss_function(torch.cat(y_pred), torch.tensor(y_true).to(device))

    return {
        'loss': loss.item(),
        'acc': positive / len(dataset)
    }
