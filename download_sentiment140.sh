#!/bin/bash
curl -L -o ./sentiment140.zip\
  https://www.kaggle.com/api/v1/datasets/download/kazanova/sentiment140

unzip -o ./sentiment140.zip -d ./sentiment140