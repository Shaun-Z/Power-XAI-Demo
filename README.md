# Generate data

Run `voltage_data_gen.py`.

A folder named `data` containing images and csv files will appear.

# Train DNN

The definition is in `cnn_model.py`.

Run `train.ipynb`. The model will be saved as `./voltage_cnn_model.pth`

# Explain DNN

Run `explain.ipynb` to compute Shapley value.

Run `explain_tree.ipynb` to compute Owen value.

---

# Mean and standard deviation

Run `data_statistic.ipynb`.

# Linear regression and explain

Run `linear_explain.ipynb`.