# Requirements
Install the proper torch version for your cuda version:
```
https://pytorch.org/get-started/locally/
```

This project requires a specific version of gliner that includes additional parameters, it can be installed with pip:
```
pip install git+https://github.com/sabdoudaoura/GLiNER_masking
```
# Checkpoints 
Checkpoints can be downloaded from: 
```
https://drive.google.com/drive/folders/1Sks62HTnjhG3ykD0hg8gbSUUEFZr4Jpe
```

# Setup 
Place articles_test.json into the data directory. Unzip the model checkpoints and place the extracted directories into the models directory. 

# Inference
Run "generate_submission_preds.sh" to generate the full set of submission predictions. 