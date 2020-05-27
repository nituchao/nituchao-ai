import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def sample_fn():
    CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

    train = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv", names=CSV_COLUMN_NAMES, header=0)
    test = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv", names=CSV_COLUMN_NAMES, header=0)

    train_y = train.pop('Species')
    test_y = test.pop('Species')
    
    train_x, verify_x, train_y, verify_y = train_test_split(train, train_y, test_size=0.33, random_state=42)
    test_x = test

    return train_x, verify_x, test_x, train_y, verify_y, test_y

def input_fn(features = None, labels = None, model_stage='train', batch_size=256):
    '''An input function for Estimator's training or evaluating
    * features: pandas.core.frame.DataFrame with header
    * labels: pandas.core.series.Series
    * is_training: True or False
    * batch_size: dataset iterator's size for every training epoch
    '''

    # shuffle and repeat(onece) when is_training true
    if ('train' == model_stage or 'verify' == model_stage):
        # transform sample input to dataset
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        
        return dataset.shuffle(1000).repeat().batch(batch_size)
    elif 'test' == model_stage:
        # transform sample input to dataset
        dataset = tf.data.Dataset.from_tensor_slices((dict(features)))

        return dataset.batch(batch_size)

    else:
        raise Exception('not support model stage, which must be one of [train, verify, test]')

def model_fn(features, model_config):
    '''Build Model function f(x) for Estimator.'''

    feature_columns = []
    for key in features.keys():
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    # DNNClassifier have 2 hidden layer with hidden_units [30, 10]
    return tf.estimator.DNNClassifier(
        hidden_units=[30, 10],
        feature_columns=feature_columns,
        n_classes=3,
        config=model_config)
    
def build_fn(model_config, model_params):
    # checkout train, verify, test dataset
    train_x, verify_x, test_x, train_y, verify_y, test_y = sample_fn()

    # model build
    dnn_model = model_fn(train_x, model_config)

    # model train
    dnn_model.train(input_fn=lambda: input_fn(train_x, train_y, model_stage='train'), steps=5000)

    # model verify
    verify_result = dnn_model.evaluate(input_fn=lambda: input_fn(verify_x, verify_y, model_stage='verify'), steps=1)

    print('{}\nVerify set accuracy: {accuracy:0.3f}\n'.format('#'*100, **verify_result))

    # model test 
    predict_y = dnn_model.predict(input_fn=lambda: input_fn(test_x, model_stage='test'))

    SPECIES = ['Setosa', 'Versicolor', 'Virginica']
    for pred_dict, expec in zip(predict_y, test_y):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{}" ({:.1f}%), expected "{}"'.format(SPECIES[class_id], 100 * probability, SPECIES[expec]))

def main(_):

    # config for init model
    model_config = tf.estimator.RunConfig().replace(
        session_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':4}),
        log_step_count_steps=1000,
        save_summary_steps=1000,
        model_dir='logs/')

    # params for train model
    model_params = {
        "learning_rate": 0.001,
        "l2_reg": 0.001,
        "dropout": 0.001
    }

    # train, verify and test model
    build_fn(model_config, model_params)

    return 0

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()