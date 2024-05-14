# Linear OpenLM

This repository contains the code for [Linearizing Large Language Models](https://arxiv.org/abs/2405.06640). This is a fork of the original [OpenLM repository](https://github.com/mlfoundations/open_lm).

## How to train a linear model

See the [Run training](#run-training) section below for the original instructions on how to train a model. The only difference is that you should use `linear` models instead of the `open_lm` models. The available linear models are:
<center>

| Model Name         |
|--------------------|
| `linear_tiny`      |
| `linear_1b`        |
| `linear_7b`        |
| `mistral_7b_linear`|
| `llama2_7b_linear` |

## How to uptrain a linear model

To uptrain a linear model, you can use the same training script as for training a linear model. The only difference is that you should use the `--pretrained` flag to specify the checkpoint you want to start from. For example, to uptrain a linear model from the `checkpoint.pt` checkpoint, you can use the following command:

```bash
>>> export CUDA_VISIBLE_DEVICES=0,1,2,3
>>> torchrun --nproc-per-node 4 -m open_lm.main   \
 --model linear_tiny \
 --dataset-manifest refined_web_tokenized/manifest.jsonl \
 --train-num-samples 1_000_000 \
 --precision "amp_bfloat16" \
 --fsdp-amp \
 --fsdp-pure-bf16 \
 --workers 1 \
 --global-batch-size 9 \
 --log-every-n-steps 100 \
 --grad-clip-norm 1 \
 --data-key json.gz \
 --lr 3e-4 \
 --accum-freq 1 \
 --warmup 10 \
 --wd 0.1 \
 --beta2 0.98 \
 --epochs 10 \
 --report-to wandb \
 --wandb-project-name linear_open_lm \
 --name linear_tiny_example \
 --logs logs \
 --z-loss-coefficient 1e-4 \
 --load-not-strict \
 --pretrained checkpoint.pt
```

## How to evaluate a linear model

See the [Evaluate Model](#evaluate-model) section below for the original instructions on how to evaluate a model. The only difference is that you should use `linear` models instead of the `open_lm` models. Note that for the reference paper, we used the EleutherAI evaluation suite, which is not available in this repository.

## How to generate text from a linear model

An example of how to generate text from a linear model is shown below. The only difference is that you should use `linear` models instead of the `open_lm` models.

```bash
python scripts/generate.py \
--model linear_1b \
--checkpoint /path/to/linear_checkpoint.pt \
--input-text "Are you conscious, can you talk to me?" \
--tokenizer EleutherAI/gpt-neox-20b \
--use-cache
```

# OpenLM

This part is copied from the original OpenLM repository, only the paragraph used in the linear models are kept.
Refer to the original repository for more information.

# Quickstart
Here we'll go over a basic example where we start from a fresh install, download and preprocess some training data, and train a model.

## Setup
We require python >=3.9, and a current installation of pyTorch, as well as several other packages. The full list of requirements is contained in `requirements.txt` and can be installed in your python enviornment via
```>>> pip install -r requirements.txt```
Next, to access `open_lm` everywhere in your virtual environment, install it using pip (from within the top level github repo)
```>>> pip install --editable . ```
Some considerations:
- We like [WandB](https://wandb.ai/) and [tensorboard](https://www.tensorflow.org/tensorboard) for logging. We specify how to use these during training below.

## Process Training Data
Next you must specify a collection of tokenized data. For the purposes of this example, we will use a recent dump of english Wikipedia, available on HuggingFace. To download this locally, we've included a script located at [datapreprocess/wiki_download.py](datapreprocess/wiki_download.py). All you have to do is specify an output directory for where the raw data should be stored:
```
python datapreprocess/wiki_download.py --output-dir path/to/raw_data
```

Next we process our training data by running it through a BPE tokenizer and chunk it into chunks of appropriate length. By default we use the tokenizer attached with [GPT-NeoX-20B](https://github.com/EleutherAI/gpt-neox). To do this, use the script `datapreprocess/make_2048.py`:
```
>>> python datapreprocess/make_2048.py \
    --input-files path_to_raw_data/*.jsonl
    --output-dir preproc_data
    --num-workers 32
    --num-consumers 1
```
Where `input-files` passes all of its (possibly many) arguments through the python `glob` module, allowing for wildcards. Optionally, data can be stored in S3 by setting the environment variables: `S3_BASE`,  and passing the flag `--upload-to-s3` to the script. This saves sharded data to the given bucket with prefix of `S3_BASE`. E.g.
```
>>> export S3_BASE=preproc_data-v1/
>>> python datapreprocess/make2048.py --upload-to-s3 ... # same arguments as before
```

## Run Training
Tokenized data can now be passed to the main training script, `open_lm/main.py`. Distributed computatation is handled via `torchrun`, and hyperparameters are specified by a variety of keyword arguments. We highlight several of the most important ones here:
- `train-data`: location of the sharded tokenized training data. If locally generated and stored, this will point to a directory containing files like `preproc_data/2048-v1/0/XXXXXXX.tar`. Data are processed using the [webdataset](https://github.com/webdataset/webdataset) package where wildcards are supported like `preproc_data/2048-v1/0/{0000000..0000099}.tar` to select the first 100 .tar files.
- `model`: Which model to use. See the table below to see valid options and parameter sizes for each.
- `train-num-samples`: how many samples to use from the specified training dataset
- `name`: name of this particular training run for logging purposes
- `report-to`: if present, can be `wandb`, `tensorboard`, or `all` to stash logging information on WandB or Tensorboard.


Model choices are contained in the following table, where, for instance `11m` indicates an 11 million parameter model and `1b` indicates a 1 billion parameter model.
<center>

| Model Name    |
|---------------|
| `open_lm_11m` |
| `open_lm_25m` |
| `open_lm_87m` |
| `open_lm_160m`|
| `open_lm_411m`|
| `open_lm_830m`|
| `open_lm_1b`  |
| `open_lm_3b`  |
| `open_lm_7b`  |

</center>

An example training run can be called as follows:
```
>>> export CUDA_VISIBLE_DEVICES=0,1,2,3
>>> torchrun --nproc-per-node 4 -m open_lm.main   \
 --model open_lm_3b \
 --train-data /preproc_data/shard-{0000000..0000099}.tar \
 --train-num-samples 1000000000 \
 --workers 8 \
 --dataset-resampled \
 --precision amp_bfloat16 \
 --batch-size 8 \
 --grad-checkpointing \
 --log-every-n-steps 100 \
 --grad-clip-norm 1 \
 --data-key txt \
 --lr 3e-4 \
 --fsdp --fsdp-amp \
 --warmup 2000 \
 --wd 0.1 \
 --beta2 0.95 \
 --epochs 100 \
 --report-to wandb \
 --wandb-project-name open_lm_example \
 --name open_lm_ex_$RANDOM \
 --resume latest \
 --logs path/to/logging/dir/
```
Checkpoints and final model weights will be saved to the specified logs directory.

During training, the above command will pick shards to train on via sampling with replacement. Training can also be done by picking shards via sampling without replacement. To do this, the input dataset(s) must first be preprocessed using the following command:
```
python -m open_lm.utils.make_wds_manifest --data-dir /preproc_data/
```
This will create a file called ```manifest.jsonl``` under ```/preproc_data```. Training can then be done by sampling wihout replacement via the following example commands:
```
>>> export CUDA_VISIBLE_DEVICES=0,1,2,3
>>> torchrun --nproc-per-node 4 -m open_lm.main   \
 --model open_lm_3b \
 --dataset-manifest /preproc_data/manifest.jsonl \
 --train-num-samples 1000000000 \
 --workers 8 \
 --precision amp_bfloat16 \
 --batch-size 8 \
 --grad-checkpointing \
 --log-every-n-steps 100 \
 --grad-clip-norm 1 \
 --data-key txt \
 --lr 3e-4 \
 --fsdp --fsdp-amp \
 --warmup 2000 \
 --wd 0.1 \
 --beta2 0.95 \
 --epochs 100 \
 --report-to wandb \
 --wandb-project-name open_lm_example \
 --name open_lm_ex_$RANDOM \
 --resume latest \
 --logs path/to/logging/dir/
```

### Dataset manifest

The manifest created with `open_lm/utils/make_wds_manifest.py` is a `jsonl` file describing the dataset. Each line in this file corresponds to a shard of the dataset and is a `json` object containing two fields:

- `"shard"`: the name of a shard in the dataset.
- `"num_sequences"`: the number of sequences contained in the shards. Each sequence contains a set length of tokens.

This manifest file provides auxiliary information about the dataset, and is assumed to be found within the same directory as the shards.

## Evaluate Model
Once trained, we can evaluate the model. This requires [LLM Foundry](https://github.com/mosaicml/llm-foundry), which can be installed via `pip install llm-foundry`. Next some configurations are required to pass to the evaluator: a skeleton of these parameters is located at [eval/in_memory_hf_eval.yaml](eval/in_memory_hf_eval.yaml). Then just run the following script, making sure to point it at the checkpoint of your trained model (and it's correspending config .json file):
```
cd eval

python eval_openlm_ckpt.py \
--eval-yaml in_memory_hf_eval.yaml \
--model open_lm_1b  \
--checkpoint /path/to/openlm_checkpoint.pt
--positional_embedding_type head_rotary

```
Note that `--positional-embedding-type head_rotary` is only necessary if using the pretrained `open_lm_1b` model hosted below. See discussion in the next section about this.

## Generate Text
One can also use a trained model to generate text. This is accessible via the script located at [scripts/generate.py](scripts/generate.py). The parameters are similar to those used in evaluation:
```
cd scripts

python generate.py \
--model open_lm_1b \
--checkpoint /path/to/openlm_checkpoint.pt \
--positional-embedding-type head_rotary \
--input-text "Please give me a recipe for chocolate chip cookies"
```

# Pre-trained Models

We provide the following pre-trained models:
- [`Mistral-SUPRA-7B`](https://huggingface.co/TRI-ML/mistral-supra): Our method uptraining the Mistral-7B model on 100B tokens of refined web into an RNN.
- [`Mamba-7B`](https://huggingface.co/TRI-ML/mamba-7b-rw): Our baseline model that we trained on 1.2T tokens of refined web (2 epochs) in the present repository. See the [Mamba repo](https://github.com/state-spaces/mamba) for the original code.

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Size</th>
            <th>Tokens</th>
            <th>HellaSwag</th>
            <th>PIQA</th>
            <th>WG</th>
            <th>ARC-E</th>
            <th>ARC-C</th>
            <th>MMLU</th>
            <th>Average</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>StableLM2</td>
            <td>1.6B</td>
            <td>2000</td>
            <td>69.0</td>
            <td>76.7</td>
            <td>63.6</td>
            <td>68.6</td>
            <td>38.9</td>
            <td>38.4</td>
            <td>59.2</td>
        </tr>
        <tr>
            <td>StableLM</td>
            <td>3B</td>
            <td>1000</td>
            <td>73.8</td>
            <td>79.3</td>
            <td>65.8</td>
            <td>72.1</td>
            <td>40.0</td>
            <td>44.2</td>
            <td>62.5</td>
        </tr>
        <tr>
            <td>Gemma</td>
            <td>2B</td>
            <td>2000</td>
            <td>71.4</td>
            <td>78.6</td>
            <td>64.4</td>
            <td>74.0</td>
            <td>41.5</td>
            <td>41.2</td>
            <td>61.9</td>
        </tr>
        <tr style="background-color: #f0f0f0;">
            <td>Mamba</td>
            <td>1.4B</td>
            <td>600</td>
            <td>59.0</td>
            <td>73.9</td>
            <td>61.4</td>
            <td>65.5</td>
            <td>32.9</td>
            <td>25.2</td>
            <td>53.0</td>
        </tr>
        <tr style="background-color: #f0f0f0;">
            <td>RWKV-5</td>
            <td>1.5B</td>
            <td>1100</td>
            <td>53.1</td>
            <td>71.6</td>
            <td>59.0</td>
            <td>62.2</td>
            <td>32.7</td>
            <td>26.2</td>
            <td>50.8</td>
        </tr>
        <tr style="background-color: #f0f0f0;">
            <td>Mamba</td>
            <td>2.8B</td>
            <td>600</td>
            <td><b>66.2</b></td>
            <td><b>75.8</b></td>
            <td><b>63.4</b></td>
            <td><b>69.7</b></td>
            <td><b>36.3</b></td>
            <td><b>26.3</b></td>
            <td><b>56.3</b></td>
        </tr>
        <tr>
            <td>Llama2</td>
            <td>7B</td>
            <td>2000</td>
            <td>76.0</td>
            <td>79.1</td>
            <td>69.1</td>
            <td>76.3</td>
            <td>46.3</td>
            <td>45.9</td>
            <td>65.4</td>
        </tr>
        <tr>
            <td>Gemma</td>
            <td>7B</td>
            <td>6000</td>
            <td>80.7</td>
            <td>81.9</td>
            <td>73.7</td>
            <td><b>81.1</b></td>
            <td>53.2</td>
            <td><b>62.9</b></td>
            <td>72.2</td>
        </tr>
        <tr>
            <td>Mistral</td>
            <td>7B</td>
            <td>8000(?)</td>
            <td><b>81.0</b></td>
            <td><b>82.1</b></td>
            <td><b>74.0</b></td>
            <td>80.9</td>
            <td><b>53.8</td>
            <td>62.4</td>
            <td><b>72.4</b></td>
        </tr>
        <tr style="background-color: #f0f0f0;">
            <td>RetNet</td>
            <td>6.7B</td>
            <td>200</td>
            <td>60.7</td>
            <td>75.4</td>
            <td>58.1</td>
            <td>--</td>
            <td>--</td>
            <td>--</td>
            <td>--</td>
        </tr>
        <tr style="background-color: #f0f0f0;">
            <td>RWKV-5</td>
            <td>7B</td>
            <td>1100</td>
            <td>70.9</td>
            <td>77.2</td>
            <td>67.4</td>
            <td>71.8</td>
            <td>43.6</td>
            <td>31.0</td>
            <td>60.3</td>
        </tr>
        <tr style="background-color: #f0f0f0;">
            <td>RWKV-5-1.7T</td>
            <td>7B</td>
            <td>1700</td>
            <td>73.0</td>
            <td>78.6</td>
            <td><b>72.9</b></td>
            <td>75.8</td>
            <td>45.6</td>
            <td><b>34.9</td>
            <td>63.5</td>
        </tr>
        <tr style="background-color: #f0f0f0;">
            <td>Mamba (ours)</td>
            <td>7B</td>
            <td>1200</td>
            <td><b>77.9</b></td>
            <td><b>81.0</b></td>
            <td><u>71.8</u></td>
            <td><b>77.5</b></td>
            <td><b>46.7</b></td>
            <td>33.3</td>
            <td><b>64.7</b></td>
        </tr>
        <tr style="background-color: #f0f0f0;">
            <td>Llama2-\methodname</td>
            <td>7B</td>
            <td>+20</td>
            <td>71.8</td>
            <td>78.6</td>
            <td>65.8</td>
            <td>71.1</td>
            <td>39.5</td>
            <td>24.9</td>
            <td>58.6</td>
        </tr>
        <tr style="background-color: #f0f0f0;">
            <td>Mistral-\methodname</td>
            <td>7B</td>
            <td>+20</td>
            <td>74.8</td>
            <td>80.1</td>
            <td>67.4</td>
            <td>74.6</td>
            <td>42.3</td>
            <td>28.0</td>
            <td>61.2</td>
        </tr>
        <tr style="background-color: #f0f0f0;">
            <td>Mistral-\methodname</td>
            <td>7B</td>
            <td>+100</td>
            <td><u>77.1</u></td>
            <td><u>80.4</u></td>
            <td>70.3</td>
            <td><u>75.9</u></td>
            <td><u>45.8</u
            <td><u>45.8</u></td>
            <td><u>34.2</u></td>
            <td><u>64.0</u></td>
        </tr>
    </tbody>
</table>

<p>Linear models (RNNs and SSMs) highlighted in gray. 5-shot results are used for MMLU. Norm results are used for PIQA, HellaSwag, ARC-C. RetNet results taken from RetNet paper.</p>



Citations
--------

If you use this model in your work, please use the following BibTeX citations:

```bibtex
@inproceedings{Mercat2024LinearizingLL,
  title={Linearizing Large Language Models},
  author={Jean Mercat and Igor Vasiljevic and Sedrick Scott Keh and Kushal Arora and Achal Dave and Adrien Gaidon and Thomas Kollar},
  year={2024},
  url={https://arxiv.org/abs/2405.06640}
}
```

```bibtex
@misc{open_lm,
  author = {Gururangan, Suchin and Wortsman, Mitchell and Gadre, Samir Yitzhak and Dave, Achal and Kilian, Maciej and Shi, Weijia and Mercat, Jean and Smyrnis, Georgios and Ilharco, Gabriel and Jordan, Matt and Heckel, Reinhard and Dimakis, Alex and Farhadi, Ali and Shankar, Vaishaal and Schmidt, Ludwig},
  title = {{open_lm}:  a minimal but performative language modeling (LM) repository},
  year = {2023},
  note = {GitHub repository},
  url = {https://github.com/mlfoundations/open_lm/}
}
```

