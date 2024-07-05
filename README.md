# Contrastive Knowledge Graph Error Detection

学习CAGED模型


## Requirements

1. All the required packages can be installed by running `pip install -r requirements.txt`.


2. 需要下载bert-base-uncased模型文件放入checkpoints目录中
   

3. 需要在dataset中放入节点描述文件wn18rr/support

## 生成噪声
生成语义近似噪声


python gen_anomaly.py --model_path checkpoints/bert-base-uncased --dataset wn18rr --device cpu --max_seq_length 64 --batch_size 256 --lm_lr 1e-4 --lm_label_smoothing 0.8 --num_workers 8 --pin_memory True
   
## 运行模型

Train：

`python Our_TopK%_RankingList.py --dataset "WN18RR" --mode train --anomaly_ratio 0.05 --mu 0.005 --lam 0.1`


Test：

`python Our_TopK%_RankingList.py --dataset "WN18RR" --mode test --anomaly_ratio 0.05 --mu 0.005 --lam 0.1`

