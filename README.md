# Dice Image Classification with a Convolutional Neural Network

A PyTorch image-classification project that predicts the visible face value of a die from a 28 x 28 grayscale image. The model maps each image to one of six classes and is evaluated on a separate held-out test set.

## Results

| Metric | Result |
|---|---:|
| Best validation accuracy | 92.21% |
| Best epoch | 15 |
| Test accuracy | 86.46% |
| Test samples | 96 |
| Trainable parameters | 148,134 |
| Recorded average inference time | 0.000366 seconds per image |

The selected checkpoint was the epoch with the highest validation accuracy rather than the final training epoch. Early stopping triggered at epoch 19, helping avoid retaining a weaker late-stage model.

## Dataset

The supplied data files store grayscale images in tabular form. Each row contains a label followed by 784 pixel values that are reshaped to 1 x 28 x 28 tensors.

- Six classes representing die faces 1 through 6
- 384 samples in data.csv for training and validation
- Stratified 80/20 training-validation split: 307 training and 77 validation images
- 96 samples in the separate test.csv evaluation set
- Pixel values normalized from 0-255 to 0-1

The repository does not currently document the original dataset source or licence. Confirm those details before redistributing or reusing the data outside this project.

## Model architecture

The CNN contains three convolutional stages:

1. Two 32-channel convolution layers, batch normalization, ReLU activations, and max pooling.
2. Two 64-channel convolution layers, batch normalization, ReLU activations, and max pooling.
3. A 128-channel convolution layer followed by adaptive average pooling.

The classifier uses dropout, a 128-to-64 fully connected layer, ReLU, and a six-output prediction layer. Training uses cross-entropy loss, AdamW with a learning rate of 0.0005, a ReduceLROnPlateau scheduler, and early stopping.

## Workflow

1. Load and validate the flattened image data.
2. Convert labels 1-6 to zero-based class indices.
3. Normalize pixels and reshape rows into image tensors.
4. Train the CNN while monitoring training and validation loss and accuracy.
5. Restore the best validation checkpoint.
6. Evaluate the selected model on the held-out test set.
7. Save the learned parameters as cnn_dice_model.pth.

## Repository contents

| File | Purpose |
|---|---|
| Code.ipynb | Complete exploration, model training, evaluation, and visualizations |
| data.csv | Training and validation image data |
| test.csv | Held-out test data |
| cnn_dice_model.pth | Saved PyTorch model state dictionary |

## Run locally

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    jupyter lab

On macOS or Linux, activate the environment with source .venv/bin/activate. Open Code.ipynb and run its cells from top to bottom. The notebook expects data.csv and test.csv in the repository root.

## Limitations and next steps

- The training set is small, so the validation estimate can vary substantially between splits.
- Test accuracy is lower than the best validation accuracy, suggesting that the validation split does not fully represent the test distribution.
- Images vary in position, rotation, scale, and brightness.
- Data augmentation, repeated stratified evaluation, or cross-validation would provide a stronger estimate of generalization.
- The recorded inference time is hardware-dependent and should not be treated as a universal benchmark.

## Reproducibility

The notebook records the full preprocessing, architecture, optimization, early-stopping, evaluation, and model-saving workflow. A fixed random state is used for the stratified data split. Exact deep-learning reproducibility can still vary by hardware and PyTorch backend.
