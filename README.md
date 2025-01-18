# Toward satisfactory public accessibility -- A crowdsourcing approach

## Overview

This project analyzes public perceptions of accessible facilities across the United States using crowdsourced online reviews for each Point-of-Interest (POI) from Google Maps. Overall, we employ multiple natural language processing techniques and regression analysis to investigate public perceptions of accessible facilities and its relationship with various socio-spatial factors.

## Abstract

As urban populations grow, the need for accessible urban design has become urgent. Traditional survey methods for assessing public perceptions of accessibility are often limited in scope. Crowdsourcing via online reviews offers a valuable alternative to understanding public perceptions, and advancements in large language models
can facilitate their use. This study uses Google Maps reviews across the United States and fine-tunes Llama 3 model with the Low-Rank Adaptation technique to analyze public sentiment on accessibility. At the POI level, most categories—restaurants, retail, hotels, and healthcare—show negative sentiments. Socio-spatial analysis
reveals that areas with higher proportions of white residents and greater socioeconomic status report more positive sentiment, while areas with more elderly, highly-educated residents exhibit more negative sentiment. Interestingly, no clear link is found between the presence of disabilities and public sentiments. Overall, this
study highlights the potential of crowdsourcing for identifying accessibility challenges and providing insights for urban planners

## Key Components

1. **Data Processing**: Scripts for filtering and preparing the dataset.
   - [data_filtering.py](https://github.com/Lingyao1219/accessible-urban/blob/main/data_preparation/review_filtering.py): Filters and cleans raw data from Google Maps reviews published by UCSD (https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/).
   - [data_filtering.py](https://github.com/Lingyao1219/accessible-urban/blob/main/data_preparation/words_list.py): The list of search terms that are used to filter potential accessible facility relevant reviews. 

2. **Training & Testing Data**:
   - [annotation](https://github.com/Lingyao1219/accessible-urban/tree/main/annotation): Contains the original annotated data
   - [train.jsonl](https://github.com/Lingyao1219/accessible-urban/blob/main/train.jsonl): Contains the annotated data for training
   - [test.jsonl](https://github.com/Lingyao1219/accessible-urban/blob/main/test.jsonl): Contains the annotated data for testing
   - [validation.ipynb](https://github.com/Lingyao1219/accessible-urban/blob/main/annotation/validation.ipynb): Displays the validation for the data annotation process

4. **Modeling**: Sentiment classifier, TF-IDF classifiers, BERT classifier, Llama3 classifier using various methods, model performance evaluation, and text processing utilities.
   - [llama3 experiments](https://github.com/Lingyao1219/accessible-urban/tree/main/llama3_experiments): contains all the relevant files to implement Llama3 model
   - [bert_classifier.ipynb](https://github.com/Lingyao1219/accessible-urban/blob/main/classifiers/bert_classifier.ipynb): Implements and trains the BERT model for perception classification.
   - [sentiment_classifiers.ipynb](https://github.com/Lingyao1219/accessible-urban/blob/main/classifiers/sentiment_classifier.ipynb): Explores various sentiment classification techniques, including RoBERTa-based sentiment.
   - [tfidf_classifiers.ipynb](https://github.com/Lingyao1219/accessible-urban/blob/main/classifiers/tfidf_classifiers.ipynb): Implements TF-IDF based classifiers for comparison.
   - [model_performance.ipynb](https://github.com/Lingyao1219/accessible-urban/blob/main/model_performance/model_performance.ipynb): Evaluates and compares performance of different models.
   - For Llama 3, we select the Llama-3-8B-Instruct as the backbone model and fine-tune it with LoRA. The setting for the best checkpoint is: We obtain 91% accuracy on the test dataset. More details can be found in [llama3_experiments](https://github.com/Lingyao1219/accessible-urban/tree/main/llama3_experiments) directory.

5. **POI-level Analysis**: Text cleaning, POI analysis, and textual analysis.
   - [poi_metrics.ipynb](https://github.com/Lingyao1219/accessible-urban/blob/main/poi_metrics.ipynb): Calculate the metrics at each 
   - [poi_analysis.ipynb](https://github.com/Lingyao1219/accessible-urban/blob/main/poi_analysis.ipynb): Analyzes patterns and trends across different POI types.
   - [clean_text.py](https://github.com/Lingyao1219/accessible-urban/blob/main/classifiers/clean_text.py): Cleans and preprocesses text data for textual analysis.
   - [stop_words.py](https://github.com/Lingyao1219/accessible-urban/blob/main/stop_words.py): Defines and manages stop words for textual analysis.
   - [semantic_analysis.ipynb](https://github.com/Lingyao1219/accessible-urban/blob/main/semantic_analysis.ipynb): Performs in-depth analysis of textual content in reviews.

6. **Regression Analysis**: Feature building scripts, regression modeling, and results analysis.
   - [Feature_build_CBSA.py](https://github.com/Lingyao1219/accessible-urban/blob/main/Feature_build_CBG.py): Builds features at the CBSA level.
   - [Feature_build_CT.py](https://github.com/Lingyao1219/accessible-urban/blob/main/Feature_build_CT.py): Builds features at the county level.
   - [Model_Regression_CBG.R](https://github.com/Lingyao1219/accessible-urban/blob/main/Model_Regression_CBG.R): R script for running regression models at the CBSA level.
   - [Model_Regression_CT.R](https://github.com/Lingyao1219/accessible-urban/blob/main/Model_Regression_CT.R): R script for running regression models at the county level.
   - [Results_analysis.py](https://github.com/Lingyao1219/accessible-urban/blob/main/Results_analysis.py): Analyzes and interprets regression results with local socioeconomic factors.


## Getting Started

Please request the processed dataset from the corresponding authors of this project before you run the code. 
1. Clone the repository
2. Install required dependencies:
   - Python 3.7+
   - R 4.0+
   - Python Libraries: ast, pandas, numpy, nltk, torch, statistics, scikit-learn, transformers, matplotlib, seaborn
3. Run data processing scripts to prepare the dataset
4. Execute modeling notebooks to train and evaluate the classifiers
5. Analyze results using the provided notebooks at the POI level
6. Perform regression analysis using the R script and Python analysis scripts

## Data Availability

The original data used in this study comes from: 
1. **Google Maps Reviews**: 
   - Source: "Google local review data" published by researchers from UCSD (https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/). 
   - Please remember to cite the original two papers.
2. **Processed Parking-related Reviews**:
   - The processed data is available upon request from the corresponding authors of this project.

Please note that use of this data must comply with the original data providers' terms of service and any applicable licensing agreements.

## Reference
```
@article{li2024toward,
  title={Toward satisfactory public accessibility: A crowdsourcing approach through online reviews to inclusive urban design},
  author={Li, Lingyao and Hu, Songhua and Dai, Yinpei and Deng, Min and Momeni, Parisa and Laverghetta, Gabriel and Fan, Lizhou and Ma, Zihui and Wang, Xi and Ma, Siyuan and others},
  journal={arXiv preprint arXiv:2409.08459},
  year={2024}
}
```
