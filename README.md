# Thrive: Early Detection of Speech Pathology
Insight Data Science Project

## Project Description
Developmental Language Disorder is a type of speech pathology. It affecrs 7% of children who often struggle at school.
Early diagnosis is key to provide them with optimal therapy but the problem is that most cases are identified at a relatively 
late age.

Thrive was developed to detect Developemental Language Disorder in children from speech recordings.
The web app is available at: [Thrive](https://thrive-webapp.herokuapp.com/) and in this repository, you can find all the necessary 
code to run the app.

---

### The data
For this project I have used a database of labelled speech recordings from Czech children who were asked to repeart utterances.
From these recordings, I extracted glottal features, prosodic features and MFCCs that I combined together and used as an input
to a machine learning pipeline.

### Modeling
After data cleaning and feature selection I trained a Logisitc Regressor to model the data and classify speech recordings.




## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
