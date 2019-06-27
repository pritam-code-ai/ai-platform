input: 
   
   pre_trained model file: 

   D80_C3.h5   

   and

   audio file in 'audio_input_data' folder. path to this audio files is- 

       audio_input_data/summary.flac  
.

output: 
  
   video file in 'video_output_data' folder. path to this video files is- 

   video_output_data/talking_face_generated_file.mp4 
 
   
.



command: 

     python talking_face_video_generation_by_python.py


     for diffrent parameter input from command-line- 

     python talking_face_video_generation_by_python.py --pre_trained_model_path_as_string "D80_C3.h5" --input_audio_file_path_as_string "audio_input_data/summary.flac"     . 

