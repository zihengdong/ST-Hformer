# ST-Hformer
ST-Hformer: A Fusion of Recurrence and Attention in a Hybrid Transformer for Spatio-Temporal Traffic Modeling

The specific flowchart of the model is shown in pic1.png.

Required Packages

pytorch>=1.11
numpy
pandas
matplotlib
pyyaml
pickle
torchinfo

The way to run the code is as follows: First, navigate to the "model" directory using the "cd" command （cd/model）.
python train.py -d <dataset> -g <gpu_id> -m <MASK>
However, masks were hardly used in the experiments.
For example, you can run `python -d PEMS08 -g 0`.

When you want to visualize the dimensionality-reduced images and heatmaps, use the following command:

python visualize.py \
--m ..data/saved_models/model_fold3_20250203_103125.pt \
--c GRU_Transformer.yaml \
--d PEMS08 \
--o .

Since I used five-fold cross - validation, you don't necessarily have to use `fold3.pt`. You can replace it with the one that yields the best results.

We used the PEMS and METRA databases. As of February 3, 2025, this model is the best - performing traffic prediction model among all the models. For details, you can search for the paper: "ST-Hformer: A Fusion of Recurrence and Attention in a Hybrid Transformer for Spatio-Temporal Traffic Modeling".
