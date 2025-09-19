Idea of the whole project is to process audio files through an audio encoder and text files with the text Tokenizer. 
For this purpose I have used wav2vec by Facebook for audio and bert model for the text
Logic of this code is to get data in the format like:
dataset{
        audio_path: "path/to/the/.wav"
        Label : "1 or 2 or 3 or 4 or 5"
        Text: "Included age and sex of the partipant"
        Audio_arrays : vector_array
        }

To start working with file; 
use the command pip install -r requirements.txt
