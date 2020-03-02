from models import TextRegModel
import time
from datetime import timedelta
import models
import argparse
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # To block TensorFlow warning


def main():
    '''
    Main program
    '''
    start_time = time.time()
    ap = argparse.ArgumentParser()

    ap.add_argument("-c", type=str, required=True, help="JSON config file")
    args = ap.parse_args()

    config_file = args.c

    text_cat = TextRegModel(config_file=config_file)

    #Read JSON config file
    with open(config_file) as f:
        config = json.load(f)


    #Training mode
    if config['do_train']=="True":
        text_cat.training()

    #Evaluation mode
    if config['do_eval']=="True":
        text_cat.evaluate(mode='evaluation')

    #Prediction mode
    if config['do_predict']=="True":
        text_cat.predict()

    print ('Elapsed Time:', str(timedelta(seconds=(time.time()-start_time))))

if __name__ == "__main__":
    main()
