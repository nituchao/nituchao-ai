

train_data = "data/census+income/adult.train"
test_data = "data/census+income/adult.test"
train = pd.read_csv(train_data, sep=",", names=["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race","sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "label"])
print(train.head(5))