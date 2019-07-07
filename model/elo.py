import pandas as pd
import numpy as np


class EloRank:

    def __init__(self, games, k=60, start_rank=1400, season_change=0.2, ot=1.0, home_boost=56):

        """
        Class to initiate and run elo model

        :param games: pd DataFrame of past games
        :param k: int - k value for use in elo change
        :param start_rank: int - rank for initialization of teams
        :param season_change: float - proportion of each teams score to regress make to mean at each season
        :param ot: float - whether ot wins are discounted
        :param home_boost: int - k boost given to the team playing at home
        """

        # initialize the parameters for the elo model
        self.games = games
        self.k = k
        self.start_rank = start_rank
        self.season_change = season_change

        # create list of teams to be included in the model
        self.teams = list(set(games["team1"]).union(set(games["team2"])))

        # init other values for model
        self.margins = np.arange(-30.5, 31.5, 1)
        self.ot_discount = ot
        self.home_boost = home_boost
        self.predictions = []
        self.ranks_columns = ["date", "team"] + [str(margin) for margin in self.margins]

        # init ranks as default
        self.ranks, self.latest_ranks = self.init_ranks

        # run the elo model for all games
        self.run()

    @staticmethod
    def margin_func(mov, delta_r):
        """
        Return the effect that the margin of victory has on K when updating elo scores

        :param mov: margin of victory in game (can be negative)
        :param delta_r: difference in team ratings (can be negative)
        :return: multiplier effect on elo change
        """
        multiplier = np.log(abs(mov) + 1) * (2.2 / (delta_r * .001 + 2.2))
        return multiplier

    @staticmethod
    def logistic_cdf(x, mean, sd=400):
        """
        Logistic distribution cumulative density function

        :param x: observation (team 1 rating)
        :param mean: mean (team 2 rating)
        :param sd: sd of logistic distribution. Choice is arbitrary
        :return: p(team 1 wins | difference in rankings)
        """

        val = 1 / (10 ** ((-1 * (x - mean)) / sd) + 1)  # logistic function
        return val

    @staticmethod
    def inverse_logistic_cdf(pexp):
        """
        Inverse of logistic distribution cumulative density function

        :param pexp: cumulative probability of observing an outcome > x
        :return: the inverse logistic distribution probability
        """

        val = (1 / pexp) - 1
        delta_r = -1 * np.log10(val) * 400
        return delta_r

    def add_to_ranks_df(self, new_ranks, date, team):
        """
        Update cls.ranks to represent new ranks aftera a game is played

        :param new_ranks: new rank for a team
        :param date: date on which new rank takes place
        :param team: team whose ranking has changed
        :return: None (outcome is modification to class property)
        """

        row = np.concatenate((np.array([date, team]), new_ranks))
        self.ranks.append(row)

    @property
    def init_ranks(self):
        """
        Build the initial rankings of teams from the distribution of win percentages over all years
        data taken from historical win percentagies at each margin of victory value, which is then translated
        to a difference in elo ranks by the the inverse_logistic_cdf function to match the historic win percentages

        :return: past ranks,latest ranks - objects to be used to store each teams current/past rank as the model is run
        """

        # get historical win percentages at each margin. Extracted from historic data
        win_props = [(len(self.games[self.games["team1_score"] - self.games["team2_score"] > margin]) / len(self.games))
                     for margin in self.margins]
        win_props = np.array(win_props, dtype="float")
        # plug in historical win percentages to inverse cdf of logistic distribution
        # to find corresponding delta(Ratings) for each win margin
        win_props[win_props == 1.0] = 1.0 - 1e-6
        win_props[win_props == 0.0] = 0.0 + 1e-6
        delta_rs = self.inverse_logistic_cdf(win_props) / 2

        # init starter_rank +- the advantaged/disadvantaged elo for every margin
        default_rank = delta_rs + self.start_rank
        self.default_rank = default_rank

        # init rank dictionary for latest ranks
        latest_ranks = {team: default_rank for team in self.teams}
        ranks = [self.ranks_columns]

        for team in self.teams:
            # append a row for each team's first game
            date = self.games.loc[self.games[(self.games["team1"] == team) |
                                             (self.games["team2"] == team)]["date"].idxmin(), ["date"]].values[0]
            row = np.concatenate((np.array([date, team]), default_rank))
            ranks.append(row)

        return ranks, latest_ranks

    def home_exp(self, team1, team2, neutral):
        """

        Output the probability that team1 wins a game given rank difference using the logistic cdf

        :param team1: elo rating for team 1
        :param team2: elo rating for team 2
        :param neutral: Boolean if the game is played on neutral ground
        :return: P(team 1 wins | rank differences)
        """
        if neutral == 1:
            return self.logistic_cdf(team1, team2)
        return self.logistic_cdf(team1 + self.home_boost, team2)

    def home_obs(self, line, mov, smooth=0):
        """

        Out;ut binary value on whether team1 margin clears the line or not

        :param line: betting line/margin - value at which it is considering a win if margin > line
        :param mov: observed margin of team1_score - team2_score
        :param smooth: Smoothing value, allowing for values between 0 and 1, to discount smaller wins
        :return: 1 or 0, if smooth = 0, on whether team1 cleared the line or not
        """
        obs = self.logistic_cdf(mov, line, smooth)
        return obs

    def elo_match(self, obs, exp, ot):
        """
        Output the elo score change according to expected and observed outcome for the home team

        :param obs: Observed outcome for home team (0 or 1)
        :param exp: Expected outcome for home team
        :param ot: Whether the game was won in ot
        :return: elo change value for the home team
        """

        if ot == 1:
            return self.ot_discount * self.k * (obs - exp)

        return self.k * (obs - exp)

    def regress(self, date):
        """
        Regress ranks back to the mean (default_rank value) by the season_change

        :param date: Date on which the regression is happening, to be used in updating the latest_ranks dict
        :return: None
        """

        # this function is called at a certain point in time
        # so you should only regress scores for teams that exist at this point in time
        already_ranked_teams = [row[1] for row in self.ranks]

        # regress and update the ranks objects
        for team in self.teams:
            if already_ranked_teams.count(team) > 1:
                self.latest_ranks[team] = (1 - self.season_change) * self.latest_ranks[team] + \
                                          self.season_change * self.default_rank
                self.add_to_ranks_df(self.latest_ranks[team], date, team)

    def run(self):
        """

        Run the entire elo model on the games object

        """

        current_season = min(self.games['season'])  # first season
        print(f"Starting season: {current_season}")

        # loop through games and "run" the season.
        # vectoriizing this part wouldn't work as ranks are temporally linked -
        # each teams rank is the result of the previous game they played, so needs to be run stepping through time
        # aka, stepping through each game
        for index, game in self.games.iterrows():

            # if it's a new season, update the season and regress scores
            if game["season"] != current_season:
                current_season = game["season"]
                print(f"Starting season: {current_season}")
                self.regress(game["date"])
            team1 = game["team1"]
            team2 = game["team2"]

            # retrieve current ranks
            team1_rank = self.latest_ranks[team1]
            team2_rank = np.flipud(self.latest_ranks[team2])

            # run the elo functions
            exp = self.home_exp(team1_rank, team2_rank, game["neutral"])
            mov = game["team1_score"] - game["team2_score"]
            obs = self.home_obs(self.margins, mov)
            change = self.elo_match(obs, exp, game["ot"])

            # add margin of victory multiplier
            if game["neutral"] == 1:
                diffs = team1_rank - team2_rank
            else:
                diffs = team1_rank + self.home_boost - team2_rank
            hcap_mov = mov - (abs(self.margins) - 0.5)
            k_multipliers = self.margin_func(hcap_mov, diffs)

            # Apply k change to each teams ranks at all margin values
            change = change * k_multipliers
            team1_update = team1_rank + change
            team2_update = np.flipud(team2_rank - change)

            # add prediction to predicted dataframe (for use in evaluating model performance)
            prediction_row = np.array(
                np.concatenate((np.array([game["date"], team1, team1_rank[30], team2, team2_rank[30], mov,
                                          self.median_score(exp)]), exp)))
            self.predictions.append(prediction_row)

            # update team ranks
            self.latest_ranks[team1] = team1_update
            self.latest_ranks[team2] = team2_update
            self.add_to_ranks_df(self.latest_ranks[team1], game["date"], team1)
            self.add_to_ranks_df(self.latest_ranks[team2], game["date"], team2)

    def brier(self):
        """
        return the brier score of the predictions. Essentially the binary MSE measure

        :return: float - brier score of the model
        """

        brier = 0
        for game in self.predictions:
            # clean up some typing issues
            if not isinstance(game[5], int):
                mov = game[5].astype("int")
            else:
                mov = game[5]
            if not isinstance(game[37], float):
                homewinprob = game[5].astype("float")
            else:
                homewinprob = game[37]

            # if home team wins and it's expected
            if homewinprob > 0.50 and mov > 0:
                toadd = (homewinprob - 1) ** 2

            # if home team loses and it's expected
            elif homewinprob < 0.50 and mov < 0:
                toadd = ((1 - homewinprob) - 1) ** 2

            # if home team loses and it's not expected
            elif homewinprob > 0.50 and mov < 0:
                toadd = homewinprob ** 2

            # if home team wins and it's not expected
            else:
                toadd = (1 - homewinprob) ** 2

            brier = brier + toadd

        brier = brier / len(self.predictions)
        return brier

    def binned_results(self, bin_size, szn='all'):
        """
        Evaluate the observed vs expected percentages in each prediction bin value.
        Ex: for games predicted between 60-65% chance of winning, what percent of time
        does the favorite win. For a well calibrated model, the perentage will fall between
        the bounds of the bin (as in 63% of the time the favorite wins for games predicted between 60-65%)
        for all bins

        :param bin_size: Difference between bounds of each bin
        :param szn: Starting season to run onwards
        :return: nested lists of bins, percentages, and games in each bin
        """
        bins = range(0, 100 + bin_size, bin_size)
        bin_size = bin_size / 100
        brackets = []
        if szn != 'all':
            predictions_bins = [z for z in self.predictions if z[0] >= pd.Timestamp(str(szn))]
        else:
            predictions_bins = self.predictions

        for bin in bins[0:-1]:
            bracket = [bin, int(bin + bin_size * 100)]

            bin = bin / 100

            # subset to find number of games in each bin
            num_games = 0
            win_count = 0

            for game in predictions_bins:
                mov = int(game[5])
                game = game[7:].astype("float")

                if bin <= game[30] < bin + bin_size:
                    num_games += 1
                    # find proportion of wins
                    if game[30] > 0.5 and mov > 0:
                        win_count += 1
                    elif game[30] < 0.5 and mov < 0:
                        win_count += 1

            bracket.append(num_games)
            bracket.append(win_count)

            if num_games != 0:
                # produce win pct for this bin
                win_prop = win_count / num_games
                if bin < 0.5:
                    win_prop = 1 - win_prop
                bracket.append(win_prop)
            else:
                bracket.append(0)

            brackets.append(bracket)

        # now loop through the brackets to add one final metric, which is the num games off calibrated the bracket is
        # aka how many changed outcomes would it take for a bin to be calibrated
        off = []
        for bin in brackets:
            bin_start = bin[0] / 100
            bin_end = bin[1] / 100
            numwins = bin[3]
            numgames = bin[2]
            pct = bin[4]

            games_off = 0
            if pct >= bin_start and pct <= bin_end:
                pass

            # only run on bins with > 0 games
            if numgames > 0:
                # if calibrated, then games_off is 0
                # else in the case that the pct is above the top bound
                while pct > bin_end:
                    games_off += 1
                    numwins -= 1
                    pct = numwins / numgames
                # else in the case that pct is less than the bottom bound
                while pct < bin_end:
                    games_off += 1
                    numwins += 1
                    pct = numwins / numgames

            off.append(games_off)

        # loop through brackets and add games off for each
        [brackets[i].append(off[i]) for i in range(0, len(off))]

        return brackets

    def brier_decomposition(self, bin_size):
        """
        A modificaiton on the brier score to combine MSE with model calibration

        :param bin_size: bin size for percentage bins
        :return: brier decomposition score for cls.predicted object
        """

        calibration = 0
        refinement = 0
        # called the binned results for use in the brier decomp calculation to measure calibration to predictions
        brackets = self.binned_results(bin_size)

        for bracket in brackets:
            bin = bracket[0] / 100
            bin_size = bracket[1] / 100
            num_games = bracket[2]
            win_prop = bracket[4]

            # create the brier decomposition
            add_calibration = num_games * (bin + bin_size / 2 - win_prop) ** 2
            calibration += add_calibration

            add_refinement = num_games * (win_prop * (1 - win_prop))
            refinement += add_refinement

        return (calibration + refinement) / len(self.games)

    def median_score(self, scorelist):
        """
        From a np.array of predicted win percentages at margins, get the margin value at the median
        Serves as a de-facto predicted scoreline for each game

        :param scorelist: np.array of predicted win% at each margin
        :return: the margin value at which the predicted win % = 50%
        """
        if not isinstance(scorelist, np.ndarray):
            raise ValueError("Scorelist must be a numpy array")

        # if the scorelist has metadata (teams, ranks, etc) subset only to predictions at each margin
        if len(scorelist) > 62:
            scorelist = scorelist[7:].astype("float")
        else:
            scorelist = scorelist.astype("float")

        # get the median value from the list
        upper = np.where(scorelist > 0.5)
        upper_margin = upper[0][-1]

        return self.margins[upper_margin]

    def sorted_ranks(self):
        """
        Sort teams and ranks in descending order
        :return: sorted ranks dataframe
        """
        sorted_ranks_df = []

        for team in self.latest_ranks.keys():
            sorted_ranks_df.append([team, self.latest_ranks[team][30]])

        sorted_ranks_df = pd.DataFrame(sorted_ranks_df, columns=["team", "rank"])
        sorted_ranks_df = sorted_ranks_df.sort_values("rank", ascending=False)
        return sorted_ranks_df

    @classmethod
    def fit(cls, games):
        """
        Grid Search over hyperparameter values for building model fit

        :param games: Historic games dataframe
        :return: outcomes dataframe of model parameters and various fit scores (brier, brier_decomp)
        """

        outcomes = []

        # these ranges are determined previous runs as the rough optimal range
        ks = range(40, 70, 1)
        seasons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        ots = [1, .9, .8, .7, .6, .5]

        insample = games[games["season"] < 2017]
        outsample = games[games["season"] >= 2017]

        for k in ks:
            for season in seasons:
                for ot in ots:
                    in_model = cls(insample, k=k, ot=ot, season_change=season)
                    out_model = cls(outsample, k=k, ot=ot, season_change=season)

                    # performances
                    in_perform = in_model.performance(2005)
                    out_perform = out_model.performance(2017)

                    row = np.array([k, season, ot,
                                    in_perform['brier'], in_perform['brier_decomp'], in_perform['score_mse'],
                                    in_perform['games_off'], in_perform['games_off_pct'],
                                    out_perform['brier'], out_perform['brier_decomp'], out_perform['score_mse'],
                                    out_perform['games_off'], out_perform['games_off_pct']])
                    outcomes.append(row)

        outcomes = pd.DataFrame(outcomes, columns=['k', 'season', 'ot', 'brier_in', 'brier_d_in', 'score_mse_in',
                                                   'games_off_in', 'games_off_pct_in', 'brier_out', 'brier_d_out',
                                                   'score_mse_out', 'games_off_out', 'games_off_pct_out'])
        return outcomes

    def predict(self, upcoming):
        """
        Extrapolate most recent team ranks to predict a set of upcoming games

        :param upcoming: dataframe of upcoming games
        :return: dataframe of games and predicted winners and median score
        """
        predicted = []

        for index, game in upcoming.iterrows():
            team1 = game["team1"]
            team2 = game["team2"]

            team1_rank = self.latest_ranks[team1]
            team2_rank = np.flipud(self.latest_ranks[team2])

            exp = self.home_exp(team1_rank, team2_rank, game["neutral"])
            add_to_predicted = [game["date"], team1, team1_rank[30], team2, team2_rank[30], exp[30],
                                -self.median_score(exp)]
            predicted.append(add_to_predicted)

        return pd.DataFrame(predicted, columns=["date", "team1", "team1_rank", "team2", "team2_rank",
                                                "win_prop", "line"])

    def score_mse(self):
        """
        MSE of median score predictions

        :return: MSE
        """
        errs = []
        for game in self.predictions:
            predicted = self.median_score(game)
            mov = game[5]
            errs.append((predicted - mov) ** 2)
        return np.mean(errs)

    def performance(self, szn=2005):
        """
        Bundle several performance scores together and return as a dictionary

        :param szn: furthest back season the scores should be calculated for (ex: 2007 would be 2007 onwards)
        :return: performance dictionary with various scores
        """

        # initiate dictionary with binned results for showing calibration
        perform_dict = {'bins': self.binned_results(5, szn)}

        # various brier / mse metrics
        perform_dict['brier'] = self.brier()
        perform_dict['brier_decomp'] = self.brier_decomposition(5)
        perform_dict['score_mse'] = self.score_mse()

        # simple accuracy metrics
        perform_dict['games_correct'] = sum([x[3] for x in perform_dict['bins']])  # games predicted correctly
        perform_dict['games_incorrect'] = sum(
            [(x[2] - x[3]) for x in perform_dict['bins']])  # games predicted inccorectly
        perform_dict['accuracy'] = perform_dict['games_correct'] / (perform_dict['games_correct'] +
                                                                    perform_dict['games_incorrect'])
        # total games off correct - how many games total until a perfectly well-calibrated model
        perform_dict['games_off'] = sum([bin[-1] for bin in perform_dict['bins']])
        # games off as a portion of the total games played
        perform_dict['games_off_pct'] = perform_dict['games_off'] / (perform_dict['games_correct'] +
                                                                     perform_dict['games_incorrect'])
        return perform_dict
