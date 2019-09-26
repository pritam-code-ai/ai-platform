#!/usr/bin/env python
# coding: utf-8




from textblob import TextBlob 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import mlflow  
import os 
import argparse 





def input_text_to_blob(input_text):
    
    your_input_summary_sentence_text_data = input_text

    blob = TextBlob(your_input_summary_sentence_text_data)

    check = ""
    
    return blob




def intelligent_sentiment_checker(blob):
    
    print("\nyour input file sentiment analysis output is- \n")

    check = ""
    
    if blob.sentiment.subjectivity > 0.5:
            if blob.sentiment.polarity > 0:
                print("positive opinion")
                check = check + "positive opinion. "
                
            elif blob.sentiment.polarity == 0:
                print("neutral opinion")
                check = check + "neutral opinion. "
            else:
                print("negative opinion")
                check = check + "negative opinion. "
    else:
            print("text is fact")
            check = check + "text is fact. "
        
        
        
        

    if blob.sentiment.subjectivity < 0.5:
            if blob.sentiment.polarity > 0:
                print("positive fact")
                check = check + "positive fact. "
                
            elif blob.sentiment.polarity == 0:
                print("neutral fact")
                check = check + "neutral fact. "
                
            else:
                print("negative fact")
                check = check + "negative fact. "

    else:
            print("text is opinion")
            check = check + "text is opinion. "
        
        
        
        
    if blob.sentiment.subjectivity == 0.5:
            if blob.sentiment.polarity > 0:
                print("text is same for optinion and fact OR not opinion or fact but compound")
                check = check + "text is same for optinion and fact OR not opinion or fact but compound. "
                print("positive compound")
                check = check + "positive compound. "
            elif blob.sentiment.polarity == 0:
                print("text is same for optinion and fact OR not opinion or fact but compound")
                check = check + "text is same for optinion and fact OR not opinion or fact but compound. "
                print("neutral compound")
                check = check + "neutral compound. "
            else:
                print("text is same for optinion and fact OR not opinion or fact but compound")
                check = check + "text is same for optinion and fact OR not opinion or fact but compound. "
                print("negative compound")
                check = check + "negative compound. "
    else:
            print("text is not compound but opinion or fact") 
            check = check + "text is not compound but opinion or fact. "
    print("\n")
    
    return check
            
        




def write_combined_string_data_to_text_file(check, input_text):
    
       
    
    data_1 = "Hi, Pritam. I am a program of NLP and computer vision. "
    data_2 = "The input text file data is- . " + input_text
    data_3 = "The input text ends. As a mention, By my intelligence, I can understand that the sentiment of your input is-. Your ." + check + "Hope to meet you again. Bye."

    utterences = data_1 + data_2 + data_3 

    print(utterences)

    utterences_file_path = "utterences.txt" 

    with open(utterences_file_path, "w") as write_utterences_to_file:
        write_utterences_to_file.write(utterences)
        write_utterences_to_file.close()
        
    return utterences, utterences_file_path
    





if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()

    parser.add_argument("--text_data_file_input_path", type=str, default = "summary_file.txt", help='input a string')

    args = parser.parse_args()

    print(args.text_data_file_input_path)



    input_file_path = args.text_data_file_input_path
    input_data_file = open(input_file_path, "r")
    input_text = input_data_file.read()
    input_data_file.close()


    blob = input_text_to_blob(input_text)
 

    check = intelligent_sentiment_checker(blob)

    print("-----------------------------------------")
    
    utterences, utterences_file_path = write_combined_string_data_to_text_file(check, input_text)

    
    print("-----------------------------------------")
    
    
    
    with mlflow.start_run():
        mlflow.log_param("text_data_file_input_path", input_file_path)
        mlflow.log_artifact("utterences.txt")
