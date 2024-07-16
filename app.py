import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')
from elasticsearch import Elasticsearch
from flask import Flask, request,render_template,redirect, url_for
from flask_bootstrap import Bootstrap  # Import Bootstrap extension
from sentence_transformers import SentenceTransformer

import os 
# from dotenv import load_dotenv

app = Flask(__name__)
Bootstrap(app)  # Initialize Bootstrap extension

model = SentenceTransformer('all-mpnet-base-v2')
# model = SentenceTransformer("static\\saved_model")

#importing from .env
# address = os.getenv('ip_address')
# user_id = os.getenv('user_id')
# password = os.getenv('password')
# certificate = os.getenv('certificate')

index_name ='test_index'
 
try:
    es = Elasticsearch("https://f05f487deded494cb2a0409f3d55bfc7.es.us-east-1.aws.elastic.cloud:443",
    api_key="V2VMdXVwQUJIMXE0TDBkVkFGOVM6TFA3TFFZeWVScEcxYVRZbXBoejVDZw==")
    
    if es.ping():
        print("Succesfully connected to ElasticSearch!!")
except ConnectionError as e:
    print("Connection Error:", e)

# Defining required functions below :

def convert_score_to_percentage(score, min_score, max_score):
    '''
    This function converts documents score into percentile value
    '''
    if max_score == min_score:
        return 100.0 if score == max_score else 10.0
    return (score - min_score) / (max_score - min_score) * 100.0


def generate_query_fun2(conditions):
    '''
    This function converts  user conditions in JSON format to a query for ES
    '''
    query = {
        'query': {
            'bool': {}
        }
    }

    for operator, field_values in conditions.items():
        clause = []
        for field, values in field_values.items():
            for value in values:
                clause.append({'match': {field: value}})
        if clause:
            if operator == 'AND':
                query['query']['bool']['must'] = clause
            elif operator == 'OR':
                query['query']['bool']['should'] = clause
            elif operator == 'EXCLUDE':
                query['query']['bool']['must_not'] = clause

    return query


def search_with_condition(title, keywords, Author, abstract, body):

    '''
    This functions takes generated query and returns searched documnets based on given conditions.
    '''
    input_conditions = {
        'title': {field: [stemmer.stem(value) for value in values] for field, values in title.items()},
        'keywords': {field: [stemmer.stem(value) for value in values] for field, values in keywords.items()},
        'Author': {field: [stemmer.stem(value) for value in values] for field, values in Author.items()},
        'abstract': {field: [stemmer.stem(value) for value in values] for field, values in abstract.items()},
        'body': {field: [stemmer.stem(value) for value in values] for field, values in body.items()}
    }

    conditions = {
        'AND': {'title': input_conditions['title']['title_AND'], 'keywords': input_conditions['keywords']['keywords_AND'], 'Author': input_conditions['Author']['Author_AND'], 'abstract': input_conditions['abstract']['abstract_AND'], 'body': input_conditions['body']['body_AND']},
        'OR': {'title': input_conditions['title']['title_OR'], 'keywords': input_conditions['keywords']['keywords_OR'], 'Author': input_conditions['Author']['Author_OR'], 'abstract': input_conditions['abstract']['abstract_OR'], 'body': input_conditions['body']['body_OR']},
        'EXCLUDE': {'title': input_conditions['title']['title_EXCLUDE'], 'keywords': input_conditions['keywords']['keywords_EXCLUDE'], 'Author': input_conditions['Author']['Author_EXCLUDE'], 'abstract': input_conditions['abstract']['abstract_EXCLUDE'], 'body': input_conditions['body']['body_EXCLUDE']}
    }

    query = generate_query_fun2(conditions)

    response = es.search(index=index_name, body=query)   # using elastic search's [test index].
    scores = [doc["_score"] for doc in response['hits']['hits']]
    min_score = 0
    max_score = max(scores)

    data = []
    for i, v in enumerate(response['hits']['hits'], start=1):
        data.append({
            'Sr_no': i,
            'Percentile': str(np.round(convert_score_to_percentage(v["_score"], min_score, max_score), 2)) + '%',
            'Document_name': v['_source']['file_name'],
            'Document_title': v['_source']['raw_title']})

    df = pd.DataFrame(data)
    return df


def query_split(condition):
    '''
    This function used to avoid unwanted inputs from user-side(ex. space, comma,etc.)
    '''
    result = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', condition).split()
    return result


def contextual_search(input_keyword,model):
    '''This functions takes query and returns searched documnets based on contexual similarity. 
    We have used pretrained transformer(BERT)
    '''

    vector_of_input_keyword = model.encode(input_keyword)

    query = {"knn": {"field": "keyword_vector","query_vector": vector_of_input_keyword,"k": 5,"num_candidates": 40}}
    print(query)
    result = es.search(index=index_name, body=query) # using elastic search's [test_index_2].
    scores = [doc["_score"] for doc in result['hits']['hits']]
    min_score = 0
    max_score = max(scores)

    data = []
    for i, v in enumerate(result['hits']['hits'], start=1):
        data.append({
            'Sr_no': i,
            'Percentile': str(np.round(convert_score_to_percentage(v["_score"], min_score, max_score), 2)) + '%',
            'Document_name': v['_source']['file_name'],
            'Document_title': v['_source']['raw_title']})

    df = pd.DataFrame(data)
    return df



# creataing API below using flask

@app.route('/')  # index page UI
def index():
    return render_template('index.html')


@app.route('/ConditionalSearch', methods=['POST']) 
def conditional_model():
    try:
        title_conditions = {'title_AND': [], 'title_OR': [], 'title_EXCLUDE': []}
        keywords_conditions = {'keywords_AND': [], 'keywords_OR': [], 'keywords_EXCLUDE': []}
        Author_conditions = {'Author_AND': [], 'Author_OR': [], 'Author_EXCLUDE': []}
        abstract_conditions = {'abstract_AND': [], 'abstract_OR': [], 'abstract_EXCLUDE': []}
        body_conditions = {'body_AND': [], 'body_OR': [], 'body_EXCLUDE': []}

        for field in ['title', 'keywords', 'Author', 'abstract', 'body']:
            terms_AND = query_split(request.form[f"{field}_AND"])
            terms_OR = query_split(request.form[f"{field}_OR"])
            terms_EXCLUDE = query_split(request.form[f"{field}_EXCLUDE"])

            if field == "title":
                title_conditions['title_AND'] = terms_AND
                title_conditions['title_OR'] = terms_OR
                title_conditions['title_EXCLUDE'] = terms_EXCLUDE
            elif field == "keywords":
                keywords_conditions['keywords_AND'] = terms_AND
                keywords_conditions['keywords_OR'] = terms_OR
                keywords_conditions['keywords_EXCLUDE'] = terms_EXCLUDE
            elif field == "Author":
                Author_conditions['Author_AND'] = terms_AND
                Author_conditions['Author_OR'] = terms_OR
                Author_conditions['Author_EXCLUDE'] = terms_EXCLUDE
            elif field == "abstract":
                abstract_conditions['abstract_AND'] = terms_AND
                abstract_conditions['abstract_OR'] = terms_OR
                abstract_conditions['abstract_EXCLUDE'] = terms_EXCLUDE
            elif field == "body":
                body_conditions['body_AND'] = terms_AND
                body_conditions['body_OR'] = terms_OR
                body_conditions['body_EXCLUDE'] = terms_EXCLUDE

        if any(title_conditions.values()) or any(keywords_conditions.values()) or any(Author_conditions.values()) or any(abstract_conditions.values()) or any(body_conditions.values()):
            search_result_df = search_with_condition(title=title_conditions, keywords=keywords_conditions,
                                                Author=Author_conditions, abstract=abstract_conditions,
                                                body=body_conditions)
            search_result_df.set_index('Sr_no', inplace=True)
            search_results = search_result_df.to_dict(orient='records')
            return render_template('result.html',search_results=search_results)
        else:
            return render_template('index.html')    

    except Exception as e:
        return "Ops, No file found."
    

@app.route('/ContextualSearch', methods=['POST'])
def contextual_model():
    try:
        text_input = request.form["search_text_model1"]
        if text_input:
            search_result_df = contextual_search(model=model, input_keyword=text_input)
            
            search_result_df.set_index('Sr_no', inplace=True)
            search_results = search_result_df.to_dict(orient='records')
            return render_template('result.html',search_results=search_results)
        else:
            return render_template('index.html')
        
    except Exception as e:
        return "Ops, No file found."
  


if __name__ == "__main__":
    app.run()
    # app.run(host='0.0.0.0', port=5001,debug=True)   
