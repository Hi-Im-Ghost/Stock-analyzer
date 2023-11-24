import tester
import learn

usecols = ["Date", "Open", "High", "Low", "Close"]

data = tester.check_data('bitcoin.csv', 'new_bitcoin.csv', usecols)

train_data, test_data, x_data_train, y_data_train, x_data_test, y_data_test = learn.set_data_for_learn(data, .8)

model = learn.rnn_model(x_data_train, y_data_train, x_data_test, y_data_test)

learn.draw_predict(train_data, test_data, model)
