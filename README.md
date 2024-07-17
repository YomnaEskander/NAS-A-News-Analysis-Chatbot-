# NAS: A news analysis system with a chatbot interface for news consumption

## Overview

Using several deep learning and NLP techniques such as Transformers, data preprocessing, webscarping dialog flow, flask and fullfilment, this project aims to streamline news consumption by embedding deep learning and machine learning models to analyze news articles before they reach users. The system categorizes articles by topic, highlights breaking news, generates informative headlines, and provides concise summaries of key points. This approach helps users quickly grasp essential information without reading lengthy articles.

## Features

![image](https://github.com/user-attachments/assets/cb63416d-928e-4ace-9c10-df81ee2ce607)


### Topic Modeling
- **Objective**: Identify and cluster similar words to categorize articles by topic.
- **Technologies Used**: Llama2, HDBSCAN, UMAP, SBERT, C-TI-IDF, CountVectorizer, and prompt engineering.

### Breaking News Classification
- **Objective**: Highlight important, real-time events.
- **Technologies Used**: Finetuned BERT using TensorFlow, TensorFlow Hub, TensorFlow Text, BERT, and custom preprocessing and classification layers.
#### Model Performance
- **Breaking News Classification**: Achieved high accuracy and low loss during training and evaluation.
  - Training Results: Loss: 0.0318, Accuracy: 0.9917
  - Validation Results: Loss: 0.2779, Accuracy: 0.9356
  - Evaluation Results: Loss: 0.2026, Accuracy: 0.9442
#### Dataset
- **Creation**: The custom dataset used for training the breaking news classification model was created using web scraping techniques.
- **Web Scraping**: Employed Beautiful Soup for extracting data from various news websites.
- **Kaggle**: The dataset is available on [Kaggle]([https://www.kaggle.com/](https://www.kaggle.com/datasets/yomnamuhammad/breaking-news)]). https://www.kaggle.com/datasets/yomnamuhammad/breaking-news 

### Headline Generation
- **Objective**: Create compelling and informative titles for each article.
- **Technologies Used**: Llama2, prompt engineering, 4-bit quantization, and Hugging Face transformers.

### Text Summarization
- **Objective**: Provide concise summaries of key points.
- **Technologies Used**: BERT extractive summarizer (`Summarizer`), transformers, and Hugging Face.

### Chatbot Integration
- **Objective**: Streamline news delivery to users through a friendly conversational interface.
- **Technologies Used**: Dialogflow, Telegram Flask, ngrok, and Dialogflow’s fulfillment logic.
- **Functionality**: The chatbot streams five news articles at a time, each processed by the topic modeling, breaking news classification, headline generation, and text summarization models, and presents the results to the user.
- **Knowledge Base**: Utilized Dialogflow’s knowledge base feature to allow users to ask about famous individuals and receive relevant information.

## Methods and Techniques

### Embeddings and Contextual Understanding
- Utilizes BERT’s contextual embeddings for more accurate and meaningful topic extraction compared to traditional methods like LDA.

### Dynamic Topic Discovery
- Employs HDBSCAN for discovering a varying number of topics, adapting to the underlying structure of the data.

### Flexibility and Adaptability
- Adapts to different languages and domains by using appropriate pre-trained BERT models or other transformer-based embeddings.

### Usage 
There are four main files, each corresponding to a different model, along with a fulfillment logic script. Download all four model files and the fulfillment logic script. Update the paths in the ModelIsLoading function within the fulfillment script to point to the locations of your downloaded models. Finally, execute the cells until the Flask server starts.





