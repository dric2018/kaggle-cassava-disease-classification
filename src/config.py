import os


class Config:
    wording_dir = os.path.abspath('../')
    data_dir = os.path.abspath('../data')
    train_images_dir = os.path.abspath('../data/train_images')
    test_images_dir = os.path.abspath('../data/test_images')
    train_batch_size = 32
    test_batch_size = 16
    lr = 1e-4
    num_epochs = 30
    n_folds = 5
    models_dir = os.path.join(os.path.abspath('../'), 'models')
    logs_dir = os.path.join(os.path.abspath('../working'), 'logs')
    submissions_dir = os.path.join(os.path.abspath('../'), 'submissions')
    device = 'cuda'
    img_size = 300
    n_channels = 3
    base_model = 'resnet50'
    freeze = False
    precision = 16
    num_classes = 5
