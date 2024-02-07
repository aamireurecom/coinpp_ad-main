# Time-Series Anomaly Detection with Implicit Neural Representations

This code is inspired by the official implementation of [COIN++: Neural Compression Across Modalities](https://arxiv.org/abs/2201.12904).

## Requirements

The file `requirements_cpu.txt` contains the requirements for the cpu version of the code, useful for development purpose. 
The suggested installation procedure is:
```bash
# Create a conda environment with python 3.9.13 and ipython (optional)
conda create -n coinpp_ad python=3.9.13 [ipython]

# Activate the environment
conda activate coinpp_ad

# choose one of the requirements*.txt files
pip install -r [requirements_gpu.txt|requirements_cpu.txt] 

# Install the codebase from the ./coinpp_ad directory
pip install -e . 
```

While it is possible to run most of the code without it, we *strongly* recommend using [wandb](https://wandb.ai/) for experiment, logging and storing, as this is tighly integrated with the codebase.

For analysis purpose, the code for the computation of the [affiliation F1 score metric](https://arxiv.org/pdf/2206.13167v1.pdf) is needed.
You can download it [here](https://figshare.com/articles/software/affiliation-metrics-py/19131425) and install it with:

```bash
wget https://figshare.com/ndownloader/files/35448233 -O affiliation_metrics.zip
unzip affiliation_metrics.zip -d affiliation-metrics-py
rm affiliation_metrics.zip
cd affiliation-metrics-py
pip install .
```

## Data

Before running experiments, make sure to set data paths in `data/dataset_paths.yml`. 
The usage of the [Secure Water Treatment dataset (SwAT)](https://itrust.sutd.edu.sg/testbeds/secure-water-treatment-swat/) requires a license, that can be asked on the website. 
The librispeech dataset can be downloaded freely. 


## Code structure

The code is structured as follows:
- ``main.py`` train a model. It is possible to run it with the ``@config.txt`` argument to load the parameters from the config file. The config arguments are detailed `configs/README.md`
- ``main_analysis.py``: run the analysis of trained model
- ``prepare_dump.py`` allows to prepare the dump of a trained model to make faster visualization
- ``stream_res.py`` contains the code for the [streamlit](https://streamlit.io/) app used to visualize the results 
- ``MACRO.py`` contains the global parameters of the code

Further useful scripts are:
- ``launcher_analysis.py``: run the analysis of a set of trained models


### Training

To train a model, run

```python main.py @config.txt```.

See `config.txt` and `main.py` for setting various arguments. 
A few example configs used to train the models can be found in the `configs` folder.

Note that: 
1) if using wandb, you need to change the wandb entity and project name to your own.
2) Is it possible to run wandb in offline mode (recommended for running experiments on a cluster) by setting the ``main.py`` argument to ``--wandb_mode=offline``

### Analysis

### Prepare dump

### Run streamlit app

### Complete example run

```bash
python main.py @configs/config_swat.txt
# A new folder is created in ./wandb with the name of the run. For example "./wandb/offline-run-20230515_164738-u6kypmve"

# Run the analysis on the trained model
python main_analysis.py --wandb_run_id u6kypmve -o -s --plot_metrics=0

# Prepare the dump of the trained model
python prepare_dump.py -o u6kypmve

# Run the streamlit app
streamlit run stream_res.py -- server.headless true

# Open the browser on the suggested link to see the results of the streamlit app
```

You can keep the streamlit app running on a separate terminal to explore the results while training new models.

### Kept scripts for legacy with coinpp original code

The following code has not been tested recently and might not work anymore.
It is kept for reference for useful development.

#### Storing modulations

Given the `wandb_run_path` from a trained model, store modulations using

```python store_modulations --wandb_run_path <wandb_run_path>```.

#### Evaluation

To evaluate the performance of a given modulation dataset (in terms of PSNR), run

```python evaluate.py --wandb_run_path <wandb_run_path> --modulation_dataset <modulation_dataset>```.

#### Saving reconstructions

To save reconstructions for a specific set of data points, run

```python reconstruction.py --wandb_run_path <wandb_run_path> --modulation_dataset <modulation_dataset> --data_indices 0 1 2 3```.

## License

MIT"# coinpp_ad-main" 
