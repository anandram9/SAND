import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from transformers import AutoProcessor, AutoTokenizer
from functools import partial

# I am converting the IDs extracted into strings and for label mapping

def load_labels_and_Text(excel_path): 
    df = pd.read_excel(excel_path)
    label_map = dict(zip(df['ID'].astype(str), df['Class']))
    text_map = dict(zip(df['ID'].astype(str), df.get("Text", pd.Series(df.index, dtype = str))))
    return label_map, text_map

def audio_files(main_folder, label_map): 
    audio_paths = []
    labels = []
    for subfolder in os.listdir(main_folder): 
        sub_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(sub_path):
            for file in os.listdir(sub_path): 
                if file.endswith('.wav'):
                    file_path = os.path.join(sub_path, file)
                    id_str = file.split('_')[0]
                    audio_paths.append(file_path)
                    labels.append(label_map[id_str])
    return audio_paths, labels

main_folder = "C:/Users/anandram/Desktop/Data analysis/task1/training"
excel_path = "C:/Users/anandram/Desktop/Data analysis/task1/sand_task_1.xlsx"

label_map, text_map = load_labels_and_Text(excel_path)
audio_paths, labels = audio_files(main_folder, label_map)
    

class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels, text_map, sampling_rate=8000, 
                 audio_model="facebook/wav2vec2-base-960h", text_model="google-bert/bert-base-uncased"):
        self.labels = labels
        self.audio_paths = audio_paths
        self.text_map = text_map
        self.sampling_rate = sampling_rate
        self.audio_processor = AutoProcessor.from_pretrained(audio_model)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        audio_array, sr = librosa.load(self.audio_paths[index], sr=self.sampling_rate)
        label = int(self.labels[index]) - 1  
        id_str = os.path.basename(self.audio_paths[index]).split('_')[0]
        text = self.text_map.get(id_str, "")

        return {
            "audio": audio_array,
            "label": label,
            "text": text
        }

    def dynamic_padding_func(batch, audio_processor, text_tokenizer):
        """
        Custom collate function for audio and text data
        """
        audio_arrays = [item["audio"] for item in batch]
        labels = [item["label"] for item in batch]
        texts = [item["text"] for item in batch]

        processed_audio = audio_processor(
            audio_arrays,
            sampling_rate=audio_processor.sampling_rate,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16000 * 10
        )

        processed_text = text_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        processed_batch = {
            "input_values": processed_audio["input_values"],  
            "attention_mask_audio": processed_audio.get("attention_mask", None),
            "input_ids": processed_text["input_ids"],  
            "attention_mask_text": processed_text["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long)
        }

        return processed_batch


dataset = AudioDataset(audio_paths, labels, text_map)
    
    
collate_fn = partial(AudioDataset.dynamic_padding_func, 
                     audio_processor=dataset.audio_processor, 
                     text_tokenizer=dataset.text_tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    
    
    

