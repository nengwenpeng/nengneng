import torch.optim as optim
from torch.utils.data import DataLoader
from dataload.dataload import *

from model import *
from config import Config


def oldnet(opt,net):
    if opt.old_true:
        if opt.net_path:
            net.load_state_dict(t.load(opt.net_path, map_location=lambda storage, loc: storage))
    return net


def train():
    opt = Config()
    net = Net()
    net = oldnet(opt, net)
    dataset = MyDataset(opt.train_path)

    dataloader = DataLoader(dataset=dataset,batch_size=opt.batch_size, shuffle=True)
    print(dataset.len)
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()
    # optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.99))

    for epoch in range(opt.max_epoch):
        if (epoch + 1) % opt.save_every == 0:
            name_net = 'checkpoints/net_' + str(opt.test_num) + '_' + str(int((epoch + 1) / opt.save_every)) + '.pth'
            t.save(net.state_dict(), name_net)
        init_loss = 0.0
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss1 = criterion1(outputs[0], inputs)
            loss2 = criterion2(outputs[1], labels)
            loss = 1*loss1 + 10*loss2
            print('loss1:%.4f'%loss1,'loss2:%.4f'%loss2)
            loss.backward()
            optimizer.step()
            # initloss = t.abs(labels - outputs[1]).sum()
            # print('sssssssssss',initloss/opt.save_every)
            # # # init_loss += initloss
            # if i % 200 == 0:    # print every 200 mini-batches
            #     print('[%d-%d] loss: %.2f'%(epoch + 1, i, loss))
        # print('[%d]loss: %.2f' % (epoch + 1,init_loss))
        # print('平均误差：', (init_loss/dataset.len))


    # 测试

    dataloader = DataLoader(dataset=MyDataset(opt.test_path),
                            batch_size=opt.batch_size, shuffle=True)
    test_loss = 0.0
    # output_labels = list[range(0)]
    for i, (inputs, labels) in enumerate(dataloader):
        outputs = net(inputs)
        # output_labels.append(outputs)
        testloss = t.abs(labels-outputs[1]).sum()
        test_loss += testloss

    loss_rate = 1-(test_loss/dataset.len)
    print('测试平均误差：', test_loss/dataset.len)
    print('测试准确率:', loss_rate)


if __name__ == '__main__':
    train()

