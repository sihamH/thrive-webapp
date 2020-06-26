<a href="https://thrive-webapp.herokuapp.com/">
    <img src="static/img/thrive.png" title="Thrive" align="right" height="70" />
</a>

# Thrive: Early Detection of Speech Pathology
Insight Data Science Project

## Project Description
Developmental Language Disorder is a type of speech pathology. It affecrs 7% of children who often struggle at school.
Early diagnosis is key to provide them with an optimal therapy but the problem is that most cases are identified at a relatively 
late age.

Thrive was developed to detect Developemental Language Disorder in children from speech recordings.
The web app is available at: [Thrive](https://thrive-webapp.herokuapp.com/) and in this repository, you can find all the necessary 
code to run the app.

### The data
For this project I have used a database of labelled speech recordings from Czech children who were asked to repeart utterances.
From these recordings, I extracted glottal features, prosodic features and MFCCs that I combined together and used as an input
to a machine learning pipeline.

### Modeling
After data cleaning and feature selection I trained a Logisitc Regressor to model the data and classify speech recordings.

## Installation

### Clone

- Clone this repo to your local machine using 
```
git clone https://github.com/sihamH/thrive-webapp.git
```

### Dependencies

- To install dependencies run
```
pip install -r requirements.txt
```

