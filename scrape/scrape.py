from bs4 import BeautifulSoup
import requests
import pandas as pd

d1 = ['Air Force', 'Albany', 'Army', 'Bellarmine', 'Binghamton', 'Boston U.', 'Brown', 'Bryant', 'Bucknell', 'Canisius',
      'Cleveland State', 'Colgate', 'Cornell', 'Dartmouth', 'Delaware', 'Denver', 'Detroit Mercy', 'Drexel', 'Duke',
      'Fairfield', 'Furman', 'Georgetown', 'Hampton', 'Hartford', 'Harvard', 'High Point', 'Hobart', 'Hofstra',
      'Holy Cross', 'Jacksonville', 'Johns Hopkins', 'Lafayette', 'Lehigh', 'Loyola', 'Manhattan', 'Marist',
      'Marquette', 'Maryland', 'Mercer', 'Michigan', 'Monmouth', "Mount St. Mary''s", 'Navy', 'NJIT', 'North Carolina',
      'Notre Dame', 'Ohio State', 'Penn', 'Penn State', 'Princeton', 'Providence', 'Quinnipiac', 'Richmond',
      'Robert Morris', 'Rutgers', 'Sacred Heart', "Saint Joseph''s", "St Bonaventure", 'Siena', "St. John''s",
      'Stony Brook', 'Syracuse', 'Towson', 'UMass', 'UMass Lowell', 'UMBC', 'Vermont', 'Villanova', 'Virginia',
      'VMI', 'Wagner', 'Yale', "Utah"]


def scrape_link(link):
    """
    Scrape one season of scores (both upcoming and played games) from Massey Ratings and stucture as DataFrames

    :param link: web link to scrape. Currently functions with raw rating off masseyratings.com
    :return: games played dataframe, games scheduled dataframe
    """
    page = requests.get(link)

    # init parser and loop through rows
    games = []
    scheduled = []
    data = BeautifulSoup(page.content, 'html.parser')
    rows = data.find_all("pre")[0].get_text().split("\n")[:-4]
    for row in rows:

        # parse date
        date = row[0:11]
        year = date[:4]
        month = date[5:7]
        day = date[8:10]
        date_string = f"{year}-{month}-{day}"
        row = row[11:].strip()

        # parse rest to list
        row = row.strip()
        row = row.split("  ")

        # get rid of empty list values and trailing spaces
        new_row = []
        for i in range(0, len(row)):
            if row[i] != '':
                new_row.append(row[i].strip())

        # parse games that are scheduled and dont have scores
        if "Sch" in new_row[-1]:
            unplayed = True
            new_row[-1] = new_row[-1][0]
        else:
            unplayed = False

        # parse tournament and ot
        # if there is extra location information, remove it

        # get rid of gratuitous info
        if len(new_row) > 4:
            new_row = new_row[0:4]
        # if both playoff and OT
        if 'O' in new_row[-1] and 'P' in new_row[-1]:
            ot = 1
            number_ot = new_row[-1][new_row[-1].index('O') + 1]
            tourney = 1
            new_row[-1] = new_row[-1].split(" ")[0]
        # if only OT
        elif 'O' in new_row[-1] and 'P' not in new_row[-1]:
            ot = 1
            number_ot = new_row[-1][new_row[-1].index('O') + 1]
            tourney = 0
            new_row[-1] = new_row[-1].split(" ")[0]
        # if only playoff
        elif 'O' not in new_row[-1] and 'P' in new_row[-1]:
            ot = 0
            number_ot = 0
            tourney = 1
            new_row[-1] = new_row[-1].split(" ")[0]  # this strips the letter out
        # if neither
        else:
            ot = 0
            number_ot = 0
            tourney = 0

        # handle parsing of at signs for game location
        if len(new_row) != 4 and '@' in new_row[1]:
            # parse the @ symbol and clean up some of the spacing
            at_index = new_row[1].index('@')
            parsed_at = [new_row[1][0:at_index], new_row[1][at_index:]]
            parsed_cleaned_at = []
            for x in parsed_at:
                parsed_cleaned_at.append(x.strip())
            new_row = new_row[0:1] + parsed_cleaned_at + new_row[2:]

        # parse teams and scores, adding neutral if necessary

        # case logic for home and away teams
        # @ is in the row if the game is at a home stadium, else neutral
        if '@' in new_row[0]:
            team1 = new_row[0][1:]  # index at 1 to remove @ sign
            team2 = new_row[2]
            team1_score = new_row[1]
            team2_score = new_row[3]
            neutral = 0
        elif '@' in new_row[2]:
            team1 = new_row[2][1:]  # index at 1 to remove @ sign
            team2 = new_row[0]
            team1_score = new_row[3]
            team2_score = new_row[1]
            neutral = 0
        else:
            team1 = new_row[2]
            team2 = new_row[0]
            team1_score = new_row[3]
            team2_score = new_row[1]
            neutral = 1

        # append to final dictionary
        row_dict = {"date": date_string, "team1": team1, "team1_score": team1_score, "team2": team2,
                    "team2_score": team2_score, "ot": ot, "number_ot": number_ot, "season": year,
                    "neutral": neutral, "tourney": tourney}
        if unplayed:
            scheduled.append(row_dict)
        else:
            games.append(row_dict)

    # save to dataframe. could be length 0 for either
    games = pd.DataFrame(games)
    scheduled = pd.DataFrame(scheduled)

    return games, scheduled


def harmonize_team_names(games, scheduled):
    """

    function to harmonize team names between multiple data sources. Also handle quotation marks for
    sql strings

    :param games: games dataframe from Massey Ratings
    :param scheduled: scheduled dataframe from Massey Ratings
    :return: games, scheduled, with team names changed to match legacy data source
    """
    translate_dict = {"Mt St Mary's": "Mount St. Mary''s", "Boston Univ": "Boston U.",
                      "St Joseph's PA": "Saint Joseph''s", "Loyola MD": "Loyola", "Massachusetts": "UMass",
                      "MA Lowell": "UMass Lowell", "St John's": "St. John''s", "Monmouth NJ": "Monmouth",
                      "Detroit": "Detroit Mercy", "Hobart & Smith": "Hobart", "Ohio St": "Ohio State",
                      "Penn St": "Penn State", "Cleveland St": "Cleveland State", "St. John's": "St. John''s",
                      "Mount St. Mary's": "Mount St. Mary''s", "Saint Joseph's": "Saint Joseph''s"}

    for index, game in games.iterrows():
        if game["team1"] in translate_dict.keys():
            games.at[index, "team1"] = translate_dict[game["team1"]]
        if game["team2"] in translate_dict.keys():
            games.at[index, "team2"] = translate_dict[game["team2"]]

    for index, game in scheduled.iterrows():
        if game["team1"] in translate_dict.keys():
            scheduled.at[index, "team1"] = translate_dict[game["team1"]]
        if game["team2"] in translate_dict.keys():
            scheduled.at[index, "team2"] = translate_dict[game["team2"]]

    return games, scheduled


def delete_if_not_d1(games, scheduled, teams):
    """
    delete all rows where one team in the competition is not from d1, implying a scrimmage game

    :param games: games dataframe from Massey Ratings
    :param scheduled: scheduled dataframe from Massey Ratings
    :param teams: list of teams in d1
    :return: games,scheduled dataframes with games where only both teams in d1
    """

    for index, game in games.iterrows():
        if game["team1"] not in teams or game["team2"] not in teams:
            games = games.drop(index=index)

    for index, game in scheduled.iterrows():
        if game["team1"] not in teams or game["team2"] not in teams:
            scheduled = scheduled.drop(index=index)

    scheduled = scheduled.reset_index()
    del scheduled["index"]
    games = games.reset_index()
    del games["index"]
    return games, scheduled
