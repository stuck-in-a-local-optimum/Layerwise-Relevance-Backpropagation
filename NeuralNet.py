"""
# Model architecture """

class MODEL(nn.Module):
    def __init__(self, model):      
        """
            Initialize model and define last layers of fine tuned model
            
            bert: BertModel from Huggingface
            
            return: None
        """
        super(MODEL, self).__init__()
        self.activation_fns = nn.ModuleDict({
            'relu'       : torch.nn.ReLU(),
            'prelu'      : torch.nn.PReLU(),
            'elu'        : torch.nn.ELU(),
            'leaky_relu' : torch.nn.LeakyReLU()
        })

        self.bert_model = model       
        self.dense = nn.Linear(768, 128, bias=False)



        if bert_model_parameter['word_level_mean_way']==1:
          self.dense = nn.Linear(768, training_parameter['hidden_layers'][0], bias=False)

        if bert_model_parameter['word_level_mean_way']==2:
                self.register_parameter(name='word_level_emb_vertical_weights', 
                                        param=torch.nn.Parameter(torch.rand(bert_model_parameter['second_last_layer_size'],
                                                                            requires_grad=True))) 
                self.dense = nn.Linear(768, training_parameter['hidden_layers'][0], bias=False)        
        elif bert_model_parameter['word_level_mean_way']==3:
                self.register_parameter(name='word_level_emb_horizontal_weights', 
                                        param=torch.nn.Parameter(torch.rand(dataset_parameter['max_seq_len'],
                                                                            requires_grad=True)))  
                self.dense = nn.Linear(768, training_parameter['hidden_layers'][0], bias=False)    
                  
        elif bert_model_parameter['word_level_mean_way']==4:
                self.register_parameter(name='word_level_emb_batch_weights', 
                                        param=torch.nn.Parameter(torch.rand(dataset_parameter['max_seq_len'], bert_model_parameter['second_last_layer_size'],
                                                                            requires_grad=True)))  
                self.dense = nn.Linear(768, training_parameter['hidden_layers'][0], bias=False)    
                  
        elif bert_model_parameter['word_level_mean_way']==5:
                self.dense = nn.Linear(dataset_parameter['max_seq_len']*bert_model_parameter['second_last_layer_size'], training_parameter['num_hidden_layers'][0])
            


        self.deep_nn = nn.ModuleList()
        input_size = training_parameter['hidden_layers'][0]
        for i in range(1, training_parameter['num_hidden_layers']):
            self.deep_nn.add_module(f'fc{i}', nn.Linear(input_size, training_parameter['hidden_layers'][i], bias=False))
            self.deep_nn.add_module(f'activation{i}', self.activation_fns[training_parameter['activatn_choice']])
            input_size = training_parameter['hidden_layers'][i]
        self.deep_nn.add_module(f"fc{training_parameter['num_hidden_layers']}", nn.Linear(input_size, 2,  bias=False))



        # self.fc1 = nn.Linear(bert_model_parameter['second_last_layer_size'], 128, bias=False)
        # self.fc1 = nn.Linear(128, 32, bias=False)

        # self.dropout2 = nn.Dropout(0.3)      
        # self.fc2 = nn.Linear(128,2, bias=False)
        # self.fc2 = nn.Linear(32,2, bias=False)

        
    def forward(self, input_ids, attention_mask, word_level, word_level_len): 
        """
            Pass input data from all layer of model
            
            input_id                  :   (list) Encoding vectors (INT) from BertTokenizerFast
            attention_mask            :   (list) Mask vector (INT [0,1]) from Bert BertTokenizerFast
            average_adjacency_matrix  :   (list of list) Average adj matrix (float (0:1)) defined with stanza dependancy graph and its degree multiplication
            word_level                :   (list) Contain number of sub words broken from parent word (INT)
            word_level_len            :   (INT) Define the length of parent sentence without any tokenization

            
            return: (float [0,1]) Last output from fine tuned model
        """



        #freezing the bert layers
        # for param in self.model.parameters():
        #   param.requires_grad = False




        #token level embeddings
        model_output = self.bert_model(input_ids=input_ids, 
                                  attention_mask=attention_mask)
        token_level_embedding = model_output.last_hidden_state
        # word_level_embedding_flat = token_level_embedding #temporary

        #word level embeddings initialized 
        word_level_embedding = torch.zeros(input_ids.shape[0], dataset_parameter['max_seq_len'], bert_model_parameter['second_last_layer_size'])


        #iterate all text in one batch 
        for batch_no in range(0, input_ids.shape[0]):

        #copy first or starting padding
            start, end = 1, 1
            for word_break_counter in range (0, word_level_len[batch_no]):
                    start = end
                    end = start+word_level[batch_no][word_break_counter]
                    word_level_embedding[batch_no][word_break_counter] = torch.mean(token_level_embedding[batch_no][start:end], 0, True)
        word_level_embedding = word_level_embedding.to(device)

        

        if bert_model_parameter['word_level_mean_way']==1:
                dense_layer_emb = torch.mean(word_level_embedding, 1)
                dense_layer_emb = torch.flatten(dense_layer_emb, start_dim=1)
                dense_layer_emb = dense_layer_emb.to(device)


        elif bert_model_parameter['word_level_mean_way']==2:
                dense_layer_emb = word_level_embedding * self.word_level_emb_vertical_weights
                dense_layer_emb = torch.mean(dense_layer_emb, 1)
                dense_layer_emb = dense_layer_emb.to(device)




        elif bert_model_parameter['word_level_mean_way']==3:
                dense_layer_emb =  word_level_embedding.permute(0,2,1) * self.word_level_emb_horizontal_weights
                
                dense_layer_emb = torch.mean(dense_layer_emb.permute(0,2,1), 1)
                dense_layer_emb = dense_layer_emb.to(device)


                
        elif bert_model_parameter['word_level_mean_way']==4:
                dense_layer_emb = word_level_embedding * self.word_level_emb_batch_weights
                dense_layer_emb = torch.mean(dense_layer_emb, 1)
                dense_layer_emb = dense_layer_emb.to(device)

        elif bert_model_parameter['word_level_mean_way']==5:
                dense_layer_emb = torch.flatten(word_level_embedding, start_dim=1)
                dense_layer_emb = dense_layer_emb.to(device)

        output = self.dense(dense_layer_emb)

        hidden_states = []
        for layer in self.deep_nn:
            output = layer(output)
        
        output = torch.log_softmax(output, dim=1)

        word_level_embedding_flat = torch.flatten(word_level_embedding, start_dim=1)

        return output, dense_layer_emb,  word_level_embedding_flat #also returning word_level_embedding to use it in calculating relevance at word-level





    def predict(self, text):
      """
            Preprocess sentence and forward in model then 
            Apply argmax on forward function of model

            text: (str) Text/
            
            return: (int 0 or 1) argmax on last layer of model
      """
      #preprocess text
      processed_text = preprocess_text(text)
  


      input_ids, attention_mask, word_break, word_break_len = tokenize_word([processed_text])

  
      



      input_ids, attention_mask, word_break, word_break_len = torch.tensor(input_ids), torch.tensor(attention_mask),  torch.tensor(word_break), torch.tensor(word_break_len)


      #put all feature variable on device
      input_ids, attention_mask, word_break, word_break_len = input_ids.to(device), attention_mask.to(device),  word_break.to(device), word_break_len.to(device)

      #get prediction from model
      with torch.no_grad():
        pred_proba, dense_layer_emb, word_level_embeddings = self.forward(input_ids, attention_mask, word_break, word_break_len)

      #process and output result of prediction
      predictions = np.argmax(pred_proba.detach().cpu().numpy(), axis = 1)
      if predictions[0]==0:
        print("positive/normal text")
      else:
        print("negative/offensive/abusive text")
      return pred_proba.cpu().detach().numpy(), dense_layer_emb, word_level_embeddings
