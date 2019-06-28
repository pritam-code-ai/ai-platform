input: 

   input text file path: 

       input_text_data_to_summarise.txt
	   
	and  

   input number of sentences in summary output: 

       a integer number less than the input number of sentences in text file. 

output: 

   summary_file.txt 

.

command: 

     python advanced_conditioned_summarization.py


     for different parameter input from command-line- 

     python advanced_conditioned_summarization.py --input_text_file_path_as_string "input_text_data_to_summarise.txt" --condition_on_number_of_sentences_as_string_data "3"

