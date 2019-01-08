

class Config(object):
    train_path = 'data/train_data.csv'
    batch_size = 1000
    max_epoch = 20
    save_every = 10
    lr = 0.001

    test_num = 0
    use_vis = False
    env = ''

    old_true = False
    net_path = 'checkpoints/net_1_5.pth'

    test_path = 'data/test_data.csv'
