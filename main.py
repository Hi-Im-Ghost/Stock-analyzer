import tester
import learn

usecols = ["Date", "Open", "High", "Low", "Close"]

data = tester.check_data('bitcoin.csv', 'new_bitcoin.csv', usecols)

train_data, val_data, test_data, x_data_train, y_data_train, x_data_val, y_data_val, x_data_test, y_data_test = learn.set_data_for_learn(
    data, .7, .2)


modelGRU = learn.rnn_GRUmodel(x_data_train, y_data_train, x_data_val, y_data_val, x_data_test, y_data_test)
learn.draw_predict(train_data, val_data, test_data, modelGRU)


modelLSTM = learn.rnn_LSTMmodel(x_data_train, y_data_train, x_data_val, y_data_val, x_data_test, y_data_test)
learn.draw_predict(train_data, val_data, test_data, modelLSTM)