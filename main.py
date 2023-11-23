import tester
import learn

usecols = ["Date", "Open", "High", "Low", "Close"]
data = tester.check_data('bitcoin.csv', 'new_bitcoin.csv', usecols)
learn.set_data_for_learn(data, .8)
