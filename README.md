# HALO

This is the source code for reproducing the inpatient dataset experiments found in the paper "Synthesizing Extremely High Dimensional Electronic Health Records."

## Generating the Dataset
This code interfaces with the pubilc MIMIC-III ICU stay database. Before using the code, you will need to apply, complete training, and download the ADMISSIONS and DIAGNOSES_ICD tables from .<physionet.org>. From there, generate an empty directory `data/`, edit the `mimic_dir` variable in the file `build_dataset.py`, and run that file. It will generate all of the relevant data files.

## Training a Model
Next, a model can be training by creating an empt `save/` directory and running the `train_model.py` script.

## Training Baseline Models
Next, any desired baseline models may be trained by changing your working directory to `baselines/{baseline_model}` and running the corresponding `train_{baseline_model}.py` script

## Evaluating the Model(s)
Finally, the trained model and its synthetic data may be evaluated. Before beginning, create the following directory paths:
* `results/datasets`
* `results/dataset_stats/plots`
* `results/testing_stats`
* `results/synthetic_training_stats`
* `results/privacy_evaluations`

After these directories are created, first run the `test_model.py` script (along with any corresponding `test_{baseline_model}.py` in the directories from the previous section). This will generation perplexity, prediction, and synthetic dataset results. From there, you may run any other evaluation scripts (prefixed with evaluate_), making sure any references to unrun baseline models are commented out. All corresponding results will be printed and saved to pickle files.
