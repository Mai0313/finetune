#!/bin/bash

datamodel-codegen --input config.json --output ./src/_types/config.py --use-annotated --use-one-literal-as-default --snake-case-field --remove-special-field-name-prefix --use-field-description --target-python-version 3.10 --use-default --class-name FinetuneConfig
