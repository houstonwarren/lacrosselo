# lacrosselo
A margin-based elo algorithm for men's college lacrosse rankings and predictions, based on [this approach](https://arxiv.org/abs/1802.00527) developed by John Scott Moreland and Matthew Superdock. Hosted at lacrosselo.com

## about
You can read about the model on [the website](lacrosselo.com/about). This repo has 3 components:
* `model`: the EloRank class used to train and run the elo model as well as develop predictions. It also includes the Google Cloud Function code for running the model in production and updating the database.
* `scrape`: Code to scrape scores from MasseyRatings.com. Also includes the Google Cloud Function for running the scraper and saving to database.
* `server`: flask web application for serving the website.

## citations

* Massey, Kenneth. “Massey Ratings - Sports Computer Ratings, Scores, and Analysis.” Massey Ratings - Sports Computer Ratings, Scores, and Analysis, www.masseyratings.com/.
* J. Scott Moreland and Matthew C. Superdock, “Predicting outcomes for games of skill by redefining what it means to win” 	arXiv:1802.00527 [stat.ME], Feb. 2018.
* Silver, Nate. “How We Calculate NBA Elo Ratings.” FiveThirtyEight, FiveThirtyEight, 21 May 2015, fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/.
* Silver, Nate. “Introducing NFL Elo Ratings.” FiveThirtyEight, FiveThirtyEight, 4 Sept. 2014, fivethirtyeight.com/features/introducing-nfl-elo-ratings/.
