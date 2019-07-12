from flask import Flask, render_template
import db_funcs


app = Flask(__name__)


@app.route('/')
def index():
    ranks = db_funcs.get_rankings()
    return render_template("index.html",ranks=ranks)


@app.route('/ranks')
def ranks():
    ranks = db_funcs.get_rankings()
    return render_template("ranks.html",ranks=ranks)


@app.route('/games')
def games():
    upcoming = db_funcs.get_predictions()
    return render_template("games.html",upcoming=upcoming)


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/performance')
def performance():
    brackets,accuracy = db_funcs.get_performance()
    return render_template("performance.html",brackets=brackets,accuracy=accuracy)


# We only need this for local development.
if __name__ == '__main__':
    app.run()
