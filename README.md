# Stock-Price-Predictor
This is a code for stock market predictor using LSTM. It takes data upto a certain date and predicts the price of next business day.
You can visualize and predict the upcoming price for any stock/ company that you want just by doing some minor changes to the following line of code:

df=web.DataReader ('GOOGL',data_source='yahoo',start='2010-01-01',end='2020-9-17')

You can replace 'GOOGL' with anyother stock name that you want
data_source represents the site or source from where the model is getting its data to learn and predict

"start" and "end" represents the timeframe of the stock price of a particular brand to learn from
