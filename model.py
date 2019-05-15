# List the CSV columns
CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key']

#Choose which column is your label
LABEL_COLUMN = 'fare_amount'

# Set the default values for each CSV column in case there is a missing value
DEFAULTS = [[0.0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]

# Create an input function that stores your data into a dataset
def read_dataset(filename, mode, batch_size = 512):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults = DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL_COLUMN)
            return features, label
    
        # Create list of files that match pattern
        file_list = tf.gfile.Glob(filename)

        # Create dataset from file list
        dataset = tf.data.TextLineDataset(file_list).map(decode_csv)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size = 10 * batch_size)
        else:
            num_epochs = 1 # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()
    return _input_fn

# Define your feature columns
# INPUT_COLUMNS = [
#     tf.feature_column.numeric_column('pickuplon'),
#     tf.feature_column.numeric_column('pickuplat'),
#     tf.feature_column.numeric_column('dropofflat'),
#     tf.feature_column.numeric_column('dropofflon'),
#     tf.feature_column.numeric_column('passengers'),
# ]

# Create a function that will augment your feature set
def add_more_features(feats):
    # Nothing to add (yet!)
    return feats

feature_cols = add_more_features(INPUT_COLUMNS)

# Create your serving input function so that your trained model will be able to serve predictions
def serving_input_fn():
    feature_placeholders = {
        column.name: tf.placeholder(tf.float32, [None]) for column in INPUT_COLUMNS
    }

    features = feature_placeholders
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

def train_and_evaluate(frac, max_depth=5, n_estimators=100):
  import numpy as np

  # get data
  train_df, eval_df = create_dataframes(frac)
  train_x, train_y = input_fn(train_df)
  # train
  from sklearn.ensemble import RandomForestRegressor
  estimator = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=0)
  estimator.fit(train_x, train_y)
  # evaluate
  eval_x, eval_y = input_fn(eval_df)
  eval_pred = estimator.predict(eval_x)
  rmse = np.sqrt(np.mean((eval_pred-eval_y)*(eval_pred-eval_y)))
  print("Eval rmse={}".format(rmse))
  return estimator, rmse

def save_model(estimator, gcspath, name):
  from sklearn.externals import joblib
  import os, subprocess, datetime
  model = 'model.joblib'
  joblib.dump(estimator, model)
  model_path = os.path.join(gcspath, datetime.datetime.now().strftime(
    'export_%Y%m%d_%H%M%S'), model)
  subprocess.check_call(['gsutil', 'cp', model, model_path])
  return model_path