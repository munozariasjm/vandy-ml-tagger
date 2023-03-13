# Vandy - Taggers

This is a collection of jet taggers for with implementation of several taggers.

## Installation

First, clone the repository and cd into it.

```bash
git clone https://github.com/munozariasjm/vandy-ml-tagger.git
cd vandy-ml-tagger
```


To run the architecture, you need to install the following packages. We recommend using a virtual environment to avoid conflicts with other packages.

```bash
pip install -r vandy-ml-tagger/requirements.txt
pip instrall vandy-ml-tagger/
```

## Usage

To run the code, you will need to have a root file you want to convert. The root file should have the branches defined in the `vandy_taggers/vandy_taggers/utils/dataset_conversion.py` file.

There are three steps one has to take to run the code completely.

### 1. Convert the root file to pkl format

How to do this conversion is clarified in the `vandy_taggers/examples/00_conversion.ipynb` notebook. This will use a multithreaded approach to convert the root file to a .pkl file.

We have also prepared a script to do this conversion. The script is located in the `scripts/convert_dataset.py` file. The script can be run as follows:

```bash
python scripts/convert_dataset.py -c Part -i <path_to_input_folder_with_ntupleroot_files> -o <path_to_output_folder>
```

The script will convert all the root files in the input folder to pkl files in the output folder. The script will also create two kind of files `train.pkl` and `test.pkl` file in the output folder. The `train.pkl` file will contain 90% of the events and the `test.pkl` file will contain the remaining 10% of the events.

### 2. Train the model

The training of the model is done in the `vandy_taggers/examples/02_training.ipynb` notebook, while the `vandy_taggers/examples/01_training.ipynb` shows how to input data into the model. The training script is located in the `scripts/train_part.py` file. The script can be run as follows:

```bash
python scripts/train_part.py -i <path_to_input_folder_with_pkl_files> -o <path_to_output_folder> -b <batch_size> -e <number_of_epochs>
```

This will train the model and save the model in the output folder. The model will be saved as `best_model.pt` and the training history will be saved as `model_EPOCH.pt`, then they can be used to evaluate the model.

### 3. Evaluate the model

The evaluation of the model is done in the `vandy_taggers/examples/03_evaluation.ipynb` notebook. The evaluation script is located in the `scripts/evaluate_part.py` file. The script can be run as follows:

```bash
python scripts/evaluate_part.py -i <path_to_input_folder_with_pkl_files> -m <path_to_model_to_use> -o <path_to_output_folder> -b <batch_size>
```

T