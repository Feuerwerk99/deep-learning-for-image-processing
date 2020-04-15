import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from model import AlexNet
import os
import json
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = data_root + "/data_set/flower_data/"  # flower data set path

train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                     transform=data_transform["train"])  # {ImageFolder: 3306}
train_num = len(train_dataset)  # train_num = 3306
flower_list = train_dataset.class_to_idx  # class_to_idx (dict): Dict with items (class_name, class_index).
# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflowers':3, 'tulips':4}
cla_dict = dict((val, key) for key, val in flower_list.items())
# {0:'daisy', 1:'dandelion', 2:'roses', 3:'sunflowers', 4:'tulips'}

# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)  # {DataLoader: 104}

validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                        transform=data_transform["val"])  # {ImageFolder: 364}
val_num = len(validate_dataset)  # val_num = 364
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=4, shuffle=True,
                                              num_workers=0)  # {DataLoader: 91}

# test_data_iter = iter(validate_loader)  # {_SingleProcessDataLoaderIter: 91}
# test_image, test_label = test_data_iter.next()
# # shape = {Size: 4} torch.Size([4, 3, 224, 224])
# # shape = {Size: 1} 4
#
#
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
# imshow(utils.make_grid(test_image))


net = AlexNet(num_classes=5, init_weights=True)

net.to(device)
loss_function = nn.CrossEntropyLoss()
# pata = list(net.parameters())
optimizer = optim.Adam(net.parameters(), lr=0.0002)

save_path = './AlexNet.pth'
best_acc = 0.0

for epoch in range(20):
    # Sets the module in training mode.
    # This has any effect only on certain modules. See documentations of particular modules for details of their
    # behaviors in training/evaluation mode, if they are affected, e.g. Dropout, BatchNorm, etc.
    net.train()
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):  # step: 103 finally
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print(time.perf_counter() - t1)

    # Sets the module in evaluation mode.
    # This has any effect only on certain modules. See documentations of particular modules for details of their
    # behaviors in training/evaluation mode, if they are affected, e.g. Dropout, BatchNorm, etc.
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for data_test in validate_loader:
            test_images, test_labels = data_test
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_labels.to(device)).sum().item()
        accurate_test = acc / val_num
        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, acc / val_num))

print('Finished Training')
