"""# Dataloader"""

class Dataset_loader(Dataset):
    def __init__(self):
        """
            Load 
                - preprocessed data from "Raw data" and "Stanza library" section and                 
                - adjacency matrix made from dependency relation from stanza
            Process
                - tokenize the dataset into encoding, mask and labels
                - word break list and its length
                - tensor dataloader of all variables/features
            
            Max length is +2 in tokenizer because tokenizer will pad start and ending of text, we are only considering length of sentace

            !!! In-future all data related functions will be integrated in the dataloader class !!!
            return: None
        """
        # df = pd.read_csv('/content/drive/MyDrive/MIDAS/LRP/IIITD-kaggle-chllng/train-data-kaggle-challenge.csv')
        # df = df[df['language']=='Hindi']

        # df = pd.read_csv('olid-training-v1.0.tsv', sep= '\t', error_bad_lines=False)
        # df['subtask_a'] = df['subtask_a'].apply(lambda label : 1 if label=='OFF' else 0)

        df  = pd.read_csv("/content/drive/MyDrive/MIDAS/LRP/dataset_2_8_15_combined.csv")





        df = df[[dataset_parameter['sentence_column'], dataset_parameter['label_column']]]
        df[dataset_parameter['sentence_column']] = df[dataset_parameter['sentence_column']].apply(lambda x: preprocess_text(x))


        df['length'] = tokenizer_word_length(df[dataset_parameter['sentence_column']])
        df = df[df.length <dataset_parameter['max_seq_len']]



        
        text = list(df.iloc[:,0])
        label = list(df.iloc[:,1])

        
        # calculate input, attention , word break and word break length from tokenize word function
        df['input_ids'], df['attention_mask'], df['word_break'], df['word_break_len'] = tokenize_word(text)
        

        # get tokenizer input id, attention mask, word_break and word break len
        input_ids, attention_mask, word_break, word_break_len = list(df['input_ids']), list(df['attention_mask']), list(df['word_break']), list(df['word_break_len'])


                
        # to preprocess data and convert all data to torch tensor
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        word_break = torch.tensor(word_break)
        word_break_len = torch.tensor(word_break_len)
        label = torch.tensor(label)
                        
        self.length = len(label)        
        self.dataset = TensorDataset(input_ids, attention_mask, word_break, word_break_len, label)
        
    def __len__(self):
        """
            This will return length of dataset
            
            return: (int) length of dataset 
        """
        return self.length

    def __getitem__(self, id):
        """
            Give encoding, mark and label at ith index given by user as id
            
            id: (int) 
            
            return: (list) custom vector at specified index (vector, mask, labels)
        """
        return self.dataset[id] 
