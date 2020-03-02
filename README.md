# Text-Regression
Implemented in Python 3.6.8 and Tensorflow Keras.
Model is saved in .h5 format.
The label chosen for this project is "continuous_target_1"
The trained model implements Bi-LSTM architecture.

How to run the code:

1. Create virtual environment
```
virtualenv [name_of_virtual_env]
```
Example: 
```
virtualenv venv
```

Activate the virtual environment
```
source venv/bin/activate
```

2. Install the required packages
```
pip3 install -r requirements.txt
```


3. Run Exploratory Data Analysis to summarize the data
```
python3 EDA.py -c [config_file] -dir [directory_to_save_EDA_results] -i [which file from JSON config file to do EDA. must be either "train_file" or "eval_file"]
```
Example: 
```
python3 EDA.py -c config.json -dir EDA -i eval_file
```


Run the training/evaluation/prediction:
4. Set the required inputs in configuration JSON file.
```
{
    "maxlen" : [maximum sequence length. Use 72 for the pretrained model],
    "model_type" : [model type. Use "BILSTM" for the pretrained model. The current available options are: "BILSTM" and "CNN".],
    "batch_size" : [batch size for training. 128 for the pretrained model.],
    "epochs" : [number of epochs],
    "word_embedding": [word embedding file in txt format],
    "max_number_of_examples_per_class": [maximum number of examples for each class for training purpose. This is to handle imbalanced data. Write "None" if you don't want to limit your data],
    "stopwords_file" : [stopwords file in txt format],
    "train_file" : [train file in csv format],
    "eval_file" : [evaluation data in csv format],
    "predict_file" : [data to predict in csv format],
    "input_column" : [column name in data csv file that contains the text],
    "target_column" : [column name in data csv file that contains the label],
    "model_dir" : [directory where the model will be saved],
    "do_train" : [write "True" if you want to do training. ],
    "do_eval" : [write "True" if you want to do evaluation on additional dataset."],
    "do_predict" : [write "True" if you want to do prediction.]
}
```

**Follow this configuration to do evaluation on the eval file using the pretrained model**
```
{
    "maxlen" : 72,
    "model_type" : "BILSTM",
    "batch_size" : 128,
    "epochs" : 30,
    "word_embedding": "./data/glove.6B.300d.txt",
    "max_number_of_examples_per_class": "None",
    "stopwords_file" : "./data/stopwords.txt",
    "train_file" : "None",
    "eval_file" : [filename],
    "predict_file" : "None",
    "input_column" : "features",
    "target_column" : "continuous_target_1",
    "model_dir" : "./model",
    "do_train" : "False",
    "do_eval" : "True",
    "do_predict" : "False"
}
```

Other examples:

Example to do training:
```
{
    "maxlen" : 72,
    "model_type" : "BILSTM",
    "batch_size" : 64,
    "epochs" : 30,
    "word_embedding": "./data/glove.6B.300d.txt",
    "max_number_of_examples_per_class": "None",
    "stopwords_file" : "./data/stopwords.txt",
    "train_file" : "./data/train.csv",
    "eval_file" : "./data/eval.csv",
    "predict_file" : "./data/prediction.csv",
    "input_column" : "features",
    "target_column" : "continuous_target_1",
    "model_dir" : "./model",
    "do_train" : "True",
    "do_eval" : "True,
    "do_predict" : "True"
}
```

Example to do prediction only:
```
{
    "maxlen" : 72,
    "model_type" : "BILSTM",
    "batch_size" : 64,
    "epochs" : 30,
    "word_embedding": "./data/glove.6B.300d.txt",
    "max_number_of_examples_per_class": "None",
    "stopwords_file" : "./data/stopwords.txt",
    "train_file" : "None",
    "eval_file" : "None",
    "predict_file" : "./data/prediction.csv",
    "input_column" : "features",
    "target_column" : "continuous_target_1",
    "model_dir" : "./model",
    "do_train" : "False",
    "do_eval" : "False",
    "do_predict" : "True"
}
```

Extra Notes:
	1. The script will automatically split the data to train and validation set (stratified 20%) for training.
	2. If "do_train" is "True", training will be performed and there will be evaluation done on test data which is obtained from stratified 20% of training data ("train_file"). This evaluation result will be saved in 'Evaluation_Report_on_Training.txt'
	3. If "do_eval" is also "True", there will be another evaluation done on evaluation data ("eval_file"). In order to do this, provide the evaluation dataset in "eval_file". Evaluation result will be saved in 'Evaluation_Report_on_Eval_Data.txt' and 'Evaluation_Results.csv'. If you don't have additional data to do evaluation, just set this flag to "False".
	4. If you want to do evaluation without training, set "do_train" to "False" and "do_eval" to "True". Provide the evaluation dataset in "eval_file". 
	5. If you want to do prediction, set "do_predict" to "True". Provide the file to predict in "predict_file".Result will be saved in 'Prediction_Results.csv'.


5. Run the main script
```
python3 main.py -c [config_file]
```
Example: 
```
python3 main.py -c config.json
```


Additional Notes:
1. Tensorboard events file that show training and eval loss, MAE, and MSE are saved in model directory.
To load: 
```
tensorboard --logdir=[events_file_directory]
```

2. Uncertainty measurement is implemented by setting Dropout layer Training to True in model architecture. The uncertainty measurement will be done on evaluation dataset when "do_eval" is set to "True". The result will be shown as Mean and Standard Deviation for each prediction. The uncertainty is measured 50 different predictions on each example.

**Citation**
This model uncertainty measurement implementation is based on these blogs:
https://www.depends-on-the-definition.com/model-uncertainty-in-deep-learning-with-monte-carlo-dropout/
https://medium.com/comet-ml/estimating-uncertainty-in-machine-learning-models-part-2-8711c832cc15
