# Chinese-Text-Classification-Pytorch-Tuning
[![LICENSE](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://github.com/996icu/996.ICU/blob/master/LICENSE)

中文文本分类，TextCNN，TextRNN，FastText，TextRCNN，BiLSTM_Attention, DPCNN, Transformer, 基于pytorch，开箱即用。

基于ray.tune实现了对不同模型进行超参数优化的功能。简单易用。

## 环境
python 3.7  
pytorch 1.1  
tqdm  
sklearn  
tensorboardX  
**ray**

## 使用说明
第一步：安装`ray` - `pip install ray`  
第二步：选定要做超参数优化的模型: 如`TextRNN`  
第三步：根据第二步选中的模型，在run.py中设定相关超参数的search_space。具体的语法可参照[这里](https://docs.ray.io/en/latest/tune/api_docs/search_space.html)。如
```
search_space = {
    'learning_rate': tune.loguniform(1e-5, 1e-2),
    'num_epochs': tune.randint(5, 21),
    'dropout': tune.uniform(0, 0.5),
    'hidden_size': tune.randint(32, 257),
    'num_layers': tune.randint(1,3)
}
```
第四步：启动50次超参数优化实验
```
python run.py --model TextCNN --tune_param True --tune_samples 50 
```
第五步：在自动生成的实验结果文件`tune_results_.csv`中查看实验记录

---

**更多用法**
```
# 使用GPU
python run.py --model TextRNN --tune_param True --tune_gpu True

# 自定义实验结果文件后缀名
python run.py --model TextRNN --tune_param True --tune_file rnn_char

# 使用ASHA scheduler来做early stopping
python run.py --model TextRNN --tune_param True --tune_asha True

# 使用当前的超参数进行模型训练，不进行超参数优化
python run.py --model TextRNN --tune_param False
```


更多细节请参照源文档
