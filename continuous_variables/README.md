# Handling Continuous Variables

This directory contains the source code for reproducing the continuous variable experiments found in the paper "Synthesizing Extremely High Dimensional Electronic Health Records."

## Generating the Dataset
This code interfaces with the pubilc MIMIC-III ICU stay database. Before using the code, you will need to apply, complete training, and download the requisite files from <https://physionet.org>. The required files are:
* 'PATIENTS.csv'
* 'ADMISSIONS.csv'
* 'DIAGNOSES_ICD.csv'
* 'PROCEDURES_ICD.csv'
* 'PRESCRIPTIONS.csv'
* 'CHARTEVENTS.csv'

Next, you need to perform the mimic3-benchmarks preprocessing according the the repository found at <https://github.com/YerevaNN/mimic3-benchmarks>. That repository has comprehensive documentation, and it will create a series of .csv files containing lab timeseries information for each ICU stay. You just need to get through the `extract_episodes_from_subjects` step.

From there,  edit the `mimic_dir` and `timeseries_dir` variables in the file `genDatasetContinuous.py`, and run that file. It will generate all of the base data files for these experiments.

Next, according to the paper and HALO method, we need to discretize the continuous variables (lab values and inter-visit gaps) in order to feed them into our model. To do so, create a `discretized_data/` directory and run the file `discretize.py`

At this point, the discretized data and correpsonding artifacts will be available, and your dataset will be fully processed.

## Setting the Config
Depending on any dataset changes, you may need to adjust the `config.py` file according to the dataset you are using. Specifically, you may need to set `code_vocab_size` and `label_vocab_size` based on what is printed at the end of running the `genDatasetContinuous.py` file and then set `lab_vocab_size` and `continuous_vocab_size` based on what is printed at the end of running the `discretize.py` file.

## Training a Model
Next, a model can be training by creating an empt `save/` directory and running the `train_model.py` script.

## Generating Data
With this model, you are ready to create comprehensive data including continous variables. Ensure that the path `results/datasets` is created, and run the file `generate.py` followed by the file `discretized_convert.py` to convert the data back to a full continuous format in the style of the original training data before it was discretized.

Note, if you want a different amount of data rather than the size of the training dataset, set the totEHRs variable on line 93 of `generate.py`.

## Evaluating the Model(s)
Finally, the trained model and its synthetic data may be evaluated. Before beginning, make sure the path `results/dataset_stats/plots` exists. Then run the `evaluate.py` script. This will generate a series of plots showcasing a wide variety of both standard and continuous valued results.
