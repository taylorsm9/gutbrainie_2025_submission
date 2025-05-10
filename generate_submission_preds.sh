#!/bin/bash

# Get model predictions
python gliner_interface.py \
  --checkpoint_path "models/model_1" \
  --input_path "data/articles_test.json" \
  --output_path "outputs/model_1_test.json"

python gliner_interface.py \
  --checkpoint_path "models/model_2" \
  --input_path "data/articles_test.json" \
  --output_path "outputs/model_2_test.json"

python gliner_interface.py \
  --checkpoint_path "models/model_3" \
  --input_path "data/articles_test.json" \
  --output_path "outputs/model_3_test.json"

# Use learned class thresholds to trim preds
python postprocessing/threshold_class.py \
  --preds "outputs/model_1_test.json" \
  --output "outputs/model_1_test_thresholds_class.json" \
  --thresholds "postprocessing/model_1_learned_thresholds.pkl"

python postprocessing/threshold_class.py \
  --preds "outputs/model_2_test.json" \
  --output "outputs/model_2_test_thresholds_class.json" \
  --thresholds "postprocessing/model_2_learned_thresholds.pkl"

python postprocessing/threshold_class.py \
  --preds "outputs/model_3_test.json" \
  --output "outputs/model_3_test_thresholds_class.json" \
  --thresholds "postprocessing/model_3_learned_thresholds.pkl"

# Combine preds into ensemble
python postprocessing/combine_ensemble_1_preds.py \
  --recall_preds "outputs/model_1_test_thresholds_class.json" \
  --precision_preds "outputs/model_2_test_thresholds_class.json" \
  --model_3_preds "outputs/model_3_test_thresholds_class.json" \
  --output "outputs/ensemble_1_preds.json"

# Adjust preds based on rules
python postprocessing/postprocessing_rules.py \
  --preds "outputs/ensemble_1_preds.json" \
  --output "final_predictions/NLPatVCU_T61_ensemble1_ensemble1.json"

  ############ ENSEMBLE 2 ##################
# Get model predictions
python gliner_interface.py \
  --checkpoint_path "models/model_4" \
  --input_path "data/articles_test.json" \
  --output_path "outputs/model_4_test.json"

python gliner_interface.py \
  --checkpoint_path "models/model_5" \
  --input_path "data/articles_test.json" \
  --output_path "outputs/model_5_test.json"

# Use learned class thresholds to trim preds
python postprocessing/threshold_class.py \
  --preds "outputs/model_4_test.json" \
  --output "outputs/model_4_test_thresholds_class.json" \
  --thresholds "postprocessing/model_4_learned_thresholds.pkl"

python postprocessing/threshold_class.py \
  --preds "outputs/model_5_test.json" \
  --output "outputs/model_5_test_thresholds_class.json" \
  --thresholds "postprocessing/model_5_learned_thresholds.pkl"

# Combine preds into ensemble
python postprocessing/combine_ensemble_2_preds.py \
  --model_4_preds "outputs/model_4_test_thresholds_class.json" \
  --model_5_preds "outputs/model_5_test_thresholds_class.json" \
  --output "outputs/ensemble_2_preds.json"

# Adjust preds based on rules
python postprocessing/postprocessing_rules.py \
  --preds "outputs/ensemble_2_preds.json" \
  --output "final_predictions/NLPatVCU_T61_ensemble2_ensemble2.json"

############# ENSEMBLE 3 ##################

## Combine preds into ensemble
python postprocessing/combine_ensemble_3_preds.py \
  --ensemble_1_preds "outputs/ensemble_1_preds.json" \
  --ensemble_2_preds "outputs/ensemble_2_preds.json" \
  --output "final_predictions/NLPatVCU_T61_ensemble3_ensemble3.json"

############# MODEL 4 ##################

# Get model predictions
python gliner_interface.py \
  --checkpoint_path "models/model_4" \
  --input_path "data/articles_test.json" \
  --output_path "outputs/model_4_test.json"

# Use learned class thresholds to trim preds
python postprocessing/threshold_class.py \
  --preds "outputs/model_4_test.json" \
  --output "outputs/model_4_test_thresholds_class.json" \
  --thresholds "postprocessing/model_4_learned_thresholds.pkl"

# Adjust preds based on rules
python postprocessing/postprocessing_rules.py \
  --preds "outputs/model_4_test_thresholds_class.json" \
  --output "final_predictions/NLPatVCU_T61_model4_model4.json"

############# MODEL 6 ##################
python gliner_interface.py \
  --checkpoint_path "models/model_6" \
  --input_path "data/articles_test.json" \
  --output_path "outputs/model_6_test.json"

# Use learned class thresholds to trim preds
# Intentionally reuse model 4's thresholds for model 6
python postprocessing/threshold_class.py \
  --preds "outputs/model_6_test.json" \
  --output "outputs/model_6_test_thresholds_class.json" \
  --thresholds "postprocessing/model_4_learned_thresholds.pkl"

# Adjust preds based on rules
python postprocessing/postprocessing_rules.py \
  --preds "outputs/model_6_test_thresholds_class.json" \
  --output "final_predictions/NLPatVCU_T61_model6_model6.json"

# Remove metadata to match exact format in the example submission
python postprocessing/remove_metadata.py