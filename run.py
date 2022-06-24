# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse

from ray import tune
from ray.tune.schedulers import ASHAScheduler

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--tune_param', default=False, type=bool, help='True for param tuning')
parser.add_argument('--tune_samples', default=50, type=int, help='Number of tuning experiments to run')
parser.add_argument('--tune_asha', default=True, type=bool, help='If use ASHA scheduler for early stopping')
parser.add_argument('--tune_file', default='', type=str, help='Suffix of filename for parameter tuning results')
parser.add_argument('--tune_gpu', default=False, type=int, help='Use GPU to tune parameters')
args = parser.parse_args()


search_space = {
    'learning_rate': tune.loguniform(1e-5, 1e-2),
    'num_epochs': tune.randint(5, 21),
    'dropout': tune.uniform(0, 0.5),
    'hidden_size': tune.randint(32, 257),
    'num_layers': tune.randint(1,3)
}



if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


    def experiment(tune_config):
        x = import_module('models.' + model_name)
        config = x.Config(dataset, embedding)

        if tune_config:
            for param in tune_config:
                setattr(config, param, tune_config[param])
        

        start_time = time.time()
        print("Loading data...")
        vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

        # train
        config.n_vocab = len(vocab)
        model = x.Model(config).to(config.device)
        if model_name != 'Transformer':
            init_network(model)
        print(model.parameters)

        if tune_config:
            res = train(config, model, train_iter, dev_iter, test_iter, tune_param=True)
            tune.report(metric=res)
        else:
            train(config, model, train_iter, dev_iter, test_iter, tune_param=False)


    # if tune parameters
    if args.tune_param:
        scheduler = ASHAScheduler(metric='metric', mode="max") if args.tune_asha else None
        
        analysis = tune.run(experiment, num_samples=args.tune_samples, config=search_space, resources_per_trial={'gpu':int(args.tune_gpu)},
            scheduler=scheduler,
            verbose=3)
        analysis.results_df.to_csv('tune_results_'+args.tune_file+'.csv')
    # if not tune parameters
    else:
        experiment(tune_config=None)
