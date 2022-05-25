import torch
from torch import nn, optim

epoch = 20
learning_rate = 0.01


def train_model(model, trainX, trainY):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    losses = []
    acces = []
    eval_losses = []
    eval_acces = []
    for i in range(epoch):
        train_loss = 0
        train_acc = 0
        model.train()
        for j in range(100):
            X = torch.from_numpy(trainX[i]).to(torch.float32)
            y = torch.from_numpy(trainY[i]).to(torch.float32)
            out = model(X)
            loss = loss_func(out, label)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()

            train_acc += num_correct.data
        losses.append(train_loss / len(train_dataset))
        acces.append(train_acc.float() / len(train_dataset))
        model.eval()
        eval_loss = 0
        eval_acc = 0

        for data in test_loader:
            img, label = data
            img = img.view(img.size(0), -1)
            if torch.cuda.is_available():
                img = Variable(img).cuda()
                label = Variable(label).cuda()
            else:
                img = Variable(img)
                label = Variable(label)
            out = model(img)
            loss = loss_func(out, label)
            eval_loss += loss.data * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.data
        eval_losses.append(eval_loss / len(test_dataset))
        eval_acces.append(eval_acc.float() / len(test_dataset))
        print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
              .format(i, train_loss / len(train_dataset), train_acc.float() / len(train_dataset),
                      eval_loss / len(test_dataset), eval_acc.float() / len(test_dataset)))
   
    return losses, acces, eval_losses, eval_acces

