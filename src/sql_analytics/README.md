# SQL analysis of the INODE4STATBOT samples

It covers the sample cleaning, splitting, analysing and evaluating for the project INODE4STATBOT.

## Description

The project is a part of the INODE4STATBOT project. It is a part of the data analysis of the samples collected from the INODE4STATBOT project. The samples are collected from the INODE4STATBOT project and are stored in the database. The samples are cleaned, split and analysed for the project.

## Getting Started

### Dependencies

- Pandas
- Numpy
- SQLAlchemy
- psycopg2
- statbotData (data source)

### Structure

#### Code

All ipynb files are in the the root folder. The code is written in Python > 3.8

In `utils` folder, there are some utility functions for the project.

#### Data

- `data` folder contains the data for the project.
  - `data/archiev` folder contains the decrepted data for the project.
  - `data/eval` folder contains the data to evaluate for the project.
    - `data/eval/res` contains the evaluation results for the project.
  - `data/statbot` contains the split data for the project.
    - `data/statbot/old` contains the split data for the old db schema.