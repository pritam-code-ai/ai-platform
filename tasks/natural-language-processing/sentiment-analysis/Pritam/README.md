# Sentiment Analysis

input: 

   input text file path: 

       summary_file.txt 
.

output: 

   utterences.txt 
.



command: 

    mlflow run .

    for different parameter input from command-line- 

    mlflow run . -P text_data_file_input_path=summary_file.txt