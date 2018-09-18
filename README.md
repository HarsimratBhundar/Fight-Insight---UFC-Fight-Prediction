# UFC-Fight-Prediction-Deep-Learning

Dash web app that uses a deep learning approach to predict UFC fight outcomes with an accruacy around 70%.

I wrote a webscraper to exctract over 3000 vectors for fighter data (fighter metrics such as Striking Accuracy or Significant Strikes/min) and close to 1000 for fight data. 
Then using a keras model I implemented a deep learning model to predict fighter outcome based on the difference in fighter metrics of two fighters.
The final portion includes a dash web app that allows the users two pick two fighters, and displays their metrics as well as the predicted outcome of their fight.
