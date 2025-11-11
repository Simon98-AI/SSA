# Social-like Transformer (SSA)
This is an official implementation for AAAI 2024 paper ["TRAR: Routing the Attention Spans in Transformers for Visual Question Answering"](https://ojs.aaai.org/index.php/AAAI/article/view/27977). It currently includes the code for training on **VQA2.0** and **CLEVR** dataset. Our SSA model for REC task is coming soon.

## Updates
- (2024/10/10) Release our SSA-VQA project.
- (2024/01/12) The project page for SSA is avaliable.

### Installation
- Clone this repo
```bash
git clone https://github.com/Simon98-AI/SSA.git
cd SSA
```

- Create a conda virtual environment and activate it
```bash
conda create -n ssa python=3.7 -y
conda activate ssa
```

- Install `CUDA==10.1` with `cudnn7` following the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `Pytorch==1.7.1` and `torchvision==0.8.2` with `CUDA==10.1`:
```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
```
- Install [Spacy](https://spacy.io/) and initialize the [GloVe](https://github-releases.githubusercontent.com/84940268/9f4d5680-4fed-11e9-9dd2-988cce16be55?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210815%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210815T072922Z&X-Amz-Expires=300&X-Amz-Signature=1bd1bd4fc52057d8ac9eec7720e3dd333e63c234abead471c2df720fb8f04597&X-Amz-SignedHeaders=host&actor_id=48727989&key_id=0&repo_id=84940268&response-content-disposition=attachment%3B%20filename%3Den_vectors_web_lg-2.1.0.tar.gz&response-content-type=application%2Foctet-stream) as follows:
```bash
pip install -r requirements.txt
wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
pip install en_vectors_web_lg-2.1.0.tar.gz
```

### Training
**Train model on VQA-v2 with default hyperparameters:**
```bash
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar'
```
and the training log will be seved to:
```
results/log/log_run_<VERSION>.txt
```
Args:
- `--DATASET={'vqa', 'clevr'}` to choose the task for training
- `--GPU=str`, e.g. `--GPU='2'` to train model on specific GPU device.
- `--SPLIT={'train', 'train+val', train+val+vg'}`, which combines different training datasets. The default training split is `train`.
- `--MAX_EPOCH=int` to set the total training epoch number.


**Resume Training**

Resume training from specific saved model weights
```bash
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar' --RESUME=True --CKPT_V=str --CKPT_E=int
```
- `--CKPT_V=str`: the specific checkpoint version
- `--CKPT_E=int`: the resumed epoch number

**Multi-GPU Training and Gradient Accumulation**
1. Multi-GPU Training:
Add `--GPU='0, 1, 2, 3...'` after the training scripts.
```bash
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar' --GPU='0,1,2,3'
```
The batch size on each GPU will be divided into `BATCH_SIZE/GPUs` automatically.

2. Gradient Accumulation:
Add `--ACCU=n` after the training scripts
```bash
python3 run.py --RUN='train' --DATASET='vqa' --MODEL='trar' --ACCU=2
```
This makes the optimizer accumulate gradients for `n` mini-batches and update the model weights once. `BATCH_SIZE` should be divided by `n`.

### Validation and Testing
**Warning**: The args `--MODEL` and `--DATASET` should be set to the same values as those in the training stage.

**Validate on Local Machine**
Offline evaluation only support the evaluations on the `coco_2014_val` dataset now.
1. Use saved checkpoint
```bash
python3 run.py --RUN='val' --MODEL='trar' --DATASET='{vqa, clevr}' --CKPT_V=str --CKPT_E=int
```

2. Use the absolute path
```bash
python3 run.py --RUN='val' --MODEL='trar' --DATASET='{vqa, clevr}' --CKPT_PATH=str
```

**Online Testing**
All the evaluations on the `test` dataset of VQA-v2 and CLEVR benchmarks can be achieved as follows:
```bash
python3 run.py --RUN='test' --MODEL='trar' --DATASET='{vqa, clevr}' --CKPT_V=str --CKPT_E=int
```

Result file are saved at:

`results/result_test/result_run_<CKPT_V>_<CKPT_E>.json`

You can upload the obtained result json file to [Eval AI](https://evalai.cloudcv.org/web/challenges/challenge-page/163/overview) to evaluate the scores.


## Acknowledgements
- [openvqa](https://github.com/MILVLG/openvqa)
- [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa)
- [trar](https://github.com/rentainhe/TRAR-VQA)
- 
## Citation
if our SSA is helpful for your research or you wish to refer the baseline results published here, we'd really appreciate it if you could cite this paper:
```
@inproceedings{DBLP:conf/aaai/HanHSTXN24,
  author       = {Yudong Han and
                  Yupeng Hu and
                  Xuemeng Song and
                  Haoyu Tang and
                  Mingzhu Xu and
                  Liqiang Nie},
  editor       = {Michael J. Wooldridge and
                  Jennifer G. Dy and
                  Sriraam Natarajan},
  title        = {Exploiting the Social-Like Prior in Transformer for Visual Reasoning},
  booktitle    = {Thirty-Eighth {AAAI} Conference on Artificial Intelligence, {AAAI}
                  2024, Thirty-Sixth Conference on Innovative Applications of Artificial
                  Intelligence, {IAAI} 2024, Fourteenth Symposium on Educational Advances
                  in Artificial Intelligence, {EAAI} 2014, February 20-27, 2024, Vancouver,
                  Canada},
  pages        = {2058--2066},
  publisher    = {{AAAI} Press},
  year         = {2024},
}
```
