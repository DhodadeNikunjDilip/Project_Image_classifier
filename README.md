Project Overview:
This project aims to develop a custom Convolutional Neural Network (CNN) for classifying  Cheetahs,Jaguars, Hyenas and Tigers in wildlife images. 
The goal is to enhance the efficiency and accuracy of predator population monitoring by automating the detection process, which is currently manual and time-consuming.

Directory Structure:
project_Image_classifier/
    ├── _checkpoints/
    │   └── final_weights.pth
    ├── _data/
    │   ├── img01.jpg
    │   ├── img02.jpg
    │   ├── ...
    │   └── img10.jpg
    ├── _data.py
    ├── _model.py
    ├── _train.py
    ├── _predict.py
    ├── interface.py
    └── _config.py


Key Files and Their Roles:
_config.py: Contains hyperparameters such as batch_size, epochs, resize_x, resize_y, and input_channels. These settings are imported by other files to ensure consistency.

_dataset.py: Defines the custom dataset class (TheDataset) and dataloader (the_dataloader) for handling image data, including resizing and augmentation.

_model.py: Implements the custom CNN architecture (TheModel) tailored for cheetah detection.

_train.py: Contains the training loop function (the_trainer) that trains the model using the specified loss function and optimizer.

_predict.py: Includes the inference function (the_predictor) for classifying new images and generating bounding box coordinates for detected cheetahs.

interface.py: Standardizes function and class names across the project for seamless integration with the grading program.

Data: 
The model uses a custom dataset with the following classes:
-Cheetah
-Hyena
-Jaguar
-Tiger
Dataset link: https://www.kaggle.com/datasets/iluvchicken/cheetah-jaguar-and-tiger
Model Architecture:
CNNModel(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (fc1): Linear(in_features=8192, out_features=512, bias=True)
  (fc2): Linear(in_features=512, out_features=4, bias=True)
  (dropout): Dropout(p=0.5, inplace=False)
)

Training:
-Optimizer: Adam (learning rate = 0.001)
-Loss function: CrossEntropyLoss
-Batch size: 32
-Epochs: 12
-Image size: 64×64 pixels

Training Protocol:
Epoch [1/12], Loss: 0.7421, Accuracy: 0.7319
Epoch [2/12], Loss: 0.5266, Accuracy: 0.8003
...
Epoch [12/12], Loss: 0.0661, Accuracy: 0.9764

Evaluation:

The model achieved a test accuracy of 70.00% on the evaluation dataset.

