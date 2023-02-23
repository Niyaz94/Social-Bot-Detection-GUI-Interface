#!/bin/sh

export FLASK_APP=./project/index.py
pipenv run flask --debug run -h 0.0.0.0
#pipenv run flask run -h 0.0.0.0

#waitress-serve --port 5000 project.index:app