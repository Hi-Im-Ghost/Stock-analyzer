import tester


usecols = ["Date", "Open", "High", "Low", "Close"]
tester.check_data('bitcoin.csv', 'new_bitcoin.csv', usecols)
