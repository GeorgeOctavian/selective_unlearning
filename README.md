# To Each (Textual Sequence) Its Own: Improving Memorized-Data Unlearning in Large Language Models


## Setup
Please install the dependencies
```
conda env create -f environment.yml
conda activate env
pip install -e llmu-py/
```

## Configure
There are five configuration settings in *experiments/configs* for SGA,TAU,GA,TA, and DPD. These can be used as templates to construct new configurations: point to new datasets to be unlearnt/validated; perform hyper-parameter tuning; assign GPU hardware, etc. 

## Execute
Please use the script below to run the provided configurations. 
```
./fire_experiments.sh
```

## Analyze
After setting up and collecting experimental data, the data can be aggregated from Weights and Biases and plotted/analysed via the scripts in *scripts/data_analysis*


## Dataset
We use Pile samples from: https://github.com/joeljang/knowledge-unlearning .Further information in *experiments/data/extraction*

## New Baseline (GA+Mismatch)
To train with GA + Mismatch, execute the below as per: https://github.com/kevinyaobytedance/llm_unlearn
```
python experiments/run_mismatch.py 
```
The checkpoint can the be used by firing:
```
python experiments/run_unified.py --config experiments/configs/mis_config/config_13b_16_0.json
```

