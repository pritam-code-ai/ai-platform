#!/usr/bin/env python
# coding: utf-8




import mlflow
from  mlflow.tracking import MlflowClient
import nltk
from nltk.corpus import stopwords
from heapq import nlargest 
import os 
import argparse  





def find_word_frequency(text):
    stopwords = nltk.corpus.stopwords.words('english')
    word_frequencies = {}  
    for word in nltk.word_tokenize(text):  
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
            
    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    return word_frequencies





def sentence_scores(sentence, word_frequency):
    sentence_scores = {}  
    for sent in sentence:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequency.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequency[word]
                    else:
                        sentence_scores[sent] += word_frequency[word]
    return sentence_scores





def rank(sentence_scores_neu, n):
    """ return the first n sentences with highest ranking """
    return nlargest(n, sentence_scores_neu, key=sentence_scores_neu.get)





def summarizer_function(input_sentence_as_string_to_summarise, condition_on_number_of_sentences_as_string_data):
    
    n = int(condition_on_number_of_sentences_as_string_data)
    
    
    sentence = nltk.sent_tokenize(str(input_sentence_as_string_to_summarise))
    word_frequency = find_word_frequency(str(input_sentence_as_string_to_summarise))
    sentence_scores_neu = sentence_scores(sentence, word_frequency)
    neu_summary_sentences = rank(sentence_scores_neu, n)
    summary = ' '.join(neu_summary_sentences)  
    
    return summary




def create_text_file_store_summary_output(summary):
    
    file_path="summary_file.txt" 
    with open(file_path, "w") as write_summary:
        write_summary.write(summary)
        write_summary.close()
    return file_path





def create_uri_folder_for_storing_data_in_mlflow():
    
    if not os.path.isdir("client_uri"):
        os.makedirs("client_uri")
    mlflow.tracking.set_tracking_uri("client_uri")
    mlflow.tracking.get_tracking_uri()
    




def log_and_tag_in_mlflosw_workflow(run_info_run_id, summary_file_path, summary):
    
    client.log_artifact(run_info_run_id, summary_file_path)
    client.log_param(run.info.run_id,"summary", summary)
    client.set_tag(run.info.run_id, "mlflow_tracking_code", "mlflow_tracking_code")
    





def process_in_mlflow(run_info_run_id):
    
    client.list_artifacts(run_info_run_id)
    client.list_experiments()
    client.list_run_infos("0")
    client.get_run(run_info_run_id)
    client.get_experiment_by_name("advanced_conditioned_summarization")
    client.get_experiment("0")
    client.rename_experiment("0", "advanced_conditioned_summarization_renamed")
    client.get_experiment("0")
    





if __name__ == "__main__":
    
    your_input_sentence_number_1 = "Hi, I am Pritam. I want to know about you."
    
    your_input_sentence_number_2 = "Please summarize this sentence for me."
    
    number_of_sentences = 3

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_text_file_path_as_string", type=str, default = "input_text_data_to_summarise.txt", help='input a string')
    
    parser.add_argument("--condition_on_number_of_sentences_as_string_data", type=str, default = number_of_sentences, help='input a string')

    args = parser.parse_args()

    #print(args.input_text_file_path_as_string)
    
    #print(args.condition_on_number_of_sentences_as_string_data)
   
    input_file_path = args.input_text_file_path_as_string
    input_data_file = open(input_file_path, "r")
    input_text = input_data_file.read()
    input_data_file.close()


    print("\n\n")

    print("summary output of input text file data is- ")

    print("--------------------------------------------\n")

    summary = summarizer_function(input_text, args.condition_on_number_of_sentences_as_string_data)
    
    print(summary)

    print("\n")
    
    create_text_file_store_summary_output(summary)
    
    create_uri_folder_for_storing_data_in_mlflow()
    
    client = MlflowClient()
    
    experiments = client.create_experiment("advanced_conditioned_summarization")
    
    run = client.create_run(experiments[0])
    
    summary_file_path = "summary_file.txt"
    
    run_info_run_id = run.info.run_id
    
    log_and_tag_in_mlflosw_workflow(run_info_run_id, summary_file_path, summary)
    
    process_in_mlflow(run.info.run_id)
    
    client.set_terminated(run.info.run_id)
    


