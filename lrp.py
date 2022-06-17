def my_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.array(torch.softmax(torch.tensor(x), 0))

# 'जिहाद इस्लाम का अनिवार्य कर्तव्य है ,जिसका उद्देश्य पूरे विश्व से गैर मुस्लिमों का सफाया करके इस्लामी खिलाफत कायम करना है'
# 'तुझसे बड़ा चुतीया अभी तक इस धरती पर पैदा नहीं हुआ'

z = np.array([-1, 1, 3])
np.where(z > 0, z, 1 * (np.exp(z) - 1))

#LRP
def LRP(sentence, model, lrp_variant, used_bias):
    ''' This function calculates the relevance value of each word corresponding to model's predicted output for given input sentence.
      params: sentence: input sentence
              model: trained classifier model
              lrp_varaint: (int)
              lrp_variant:  0 --> 0 LRP (for output layer)
                            1 --> gamma LRP (for lower layers)
                            2 --> epsilon LRP (for upper layers)
               used_bias: boolean, (used bias or not in fc layers of classifier)


      returns: a tuple of (inputsentence, predicted_value p, input_words_relevances value of each word in input sentence) '''
    print('sentence: ', sentence)
    print('label: ', end=' ')
    model.eval()
    output_proba, dense_layer_emb, word_level_emb = model.predict(sentence)
    # print('word_level_embeddings: ', text_emb)


    if lrp_variant==0:
      print('==================================0 LRP======================================================================')
    elif lrp_variant==1:
      print('==================================GAMMA-LRP======================================================================')
    elif lrp_variant==2:
      print('==================================EPSILON-LRP======================================================================')

      

    #Step 1: ----------------Extract layers---------------------------------------------------------------------
    # 
    layers = []
    layers.append(model.dense)
    for layer in list(model.deep_nn):
      if isinstance(layer, nn.Linear):
        layers.append(layer)


    print(layers)
    n_layers = len(layers)

    dense_layer_emb = dense_layer_emb.cpu().detach().numpy()
    dense_layer_emb = dense_layer_emb.reshape(1, -1)  #reshapping for matrix multiplication

    #initializing activation of input layer with input text_emb
    #and other layer's None
    activations = [dense_layer_emb] + [None]*n_layers


    #step2: Propagate input text_emb through each layer and store activations------------------------------------

    for layer in range(n_layers):
 
      # w = model._modules[layers[layer]].state_dict()['weight'].cpu().detach().numpy()
      w = layers[layer].state_dict()['weight'].cpu().detach().numpy()

      #if used biases in fc layers of text classifier then next layer activation = w.x+b
      if used_bias==True:
        # b = model._modules[layers[layer]].state_dict()['bias'].cpu().detach().numpy()
        b = layers[layer].state_dict()['bias'].cpu().detach().numpy()

        #if last layer then apply log_softmax() activation fn on this layer
        if layer==n_layers-1:
          # activations[layer+1] = np.log(np.array([my_softmax(activations[layer].dot(w.T+b)[0])])) #log_softmax(w.x+b)
          activations[layer+1] = np.array([my_softmax(activations[layer].dot(w.T+b)[0])]) #only softmax(w.x+b) 
          


        #if other layers then apply rely activation fn
        else:
          z = activations[layer].dot(w.T)+b
          if(training_parameter['activatn_choice']=='relu'):
            activations[layer+1] = np.maximum(0, z) #relu() 

          elif (training_parameter['activatn_choice']=='leaky_relu'):
            activations[layer+1] = np.maximum(0.01*z, z) #leaky_relu() 

          elif (training_parameter['activatn_choice']=='elu'):
            alpha = 1 #pytorch default value
            activations[layer+1] = np.where(z > 0, z, alpha* (np.exp(z) - 1)) #ELU()

          elif training_parameter['activatn_choice']=='prelu':
            alpha = model.activation_fns['prelu'].weight
            # print(alpha)
            alpha = alpha[0].detach().cpu().numpy()
            activations[layer+1] = np.where(z >= 0, z, alpha*z) #PRELU()



      #if didn't used biases in fc layers of text classifier then next layer activation = w.x only
      else:
        if layer==n_layers-1:
            # activations[layer+1] = np.log(np.array([my_softmax(activations[layer].dot(w.T)[0])])) #log_softmax(w.x+b) 
          activations[layer+1] = np.array([my_softmax(activations[layer].dot(w.T)[0])]) #only softmax(w.x+b) 
        else:
          z = activations[layer].dot(w.T)
          if(training_parameter['activatn_choice']=='relu'):
            activations[layer+1] = np.maximum(0, z) #relu() 

          elif (training_parameter['activatn_choice']=='leaky_relu'):
            activations[layer+1] = np.maximum(0.01*z, z) #leaky_relu() 

          elif (training_parameter['activatn_choice']=='elu'):
            alpha = 1 #pytorch default value
            activations[layer+1] = np.where(z > 0, z, alpha* (np.exp(z) - 1)) #ELU()

          elif training_parameter['activatn_choice']=='prelu':
            alpha = model.activation_fns['prelu'].weight
            # print('prelu alpha: ', alpha)
            alpha = alpha[0].detach().cpu().numpy()
            activations[layer+1] = np.where(z >= 0, z, alpha*z) #PRELU()
     



    output_activation = activations[-1]
    output_proba = output_activation
 
    #----------------------NOT DOING STEP-3---------------------
    max_activation = output_activation.max()        
    #step3:---In output layer, except the true class activation, set all other class's values to 0----------------
    # output_activation = [val if val == max_activation else 0 for val in output_activation]

    activations[-1] = output_activation
    activations[-1]  = np.array(activations[-1]).reshape(1, 2)   #reshaped it to 1*num_classes


      
      
      
    #step4: -------Backpropagate relevance values from output layer to input layer--------------------------------------

    #initialize output layer's relevance to its activation, rest layer's as None
    relevances = [None]*n_layers + [activations[-1]]


    # print('activations[-1]: ', activations[-1])

    # Iterate over the layers in reverse order
    for layer in range(n_layers)[::-1]:
      #getting weights of the layer
      # w = model._modules[layers[layer]]
      w = layers[layer].state_dict()['weight'].T.cpu().detach().numpy()
    # b = model._modules[layers[layer]].state_dict()['bias'].cpu().detach().numpy()
    
    # rho is a function to transform the weights
      if lrp_variant==0:      #LRP-0 RULE
        rho = lambda p: p     
        incr = lambda z: z+1e-9 
      elif lrp_variant==1:    #LRP-gamma RULE
        gamma = 0.1
        rho = lambda p: p + gamma*p.clip(min=0)

        incr = lambda z: z+1e-9
      elif lrp_variant==2:    #LRP-epsilon RULE
        epsilon = 0.1
        rho = lambda p: p
        incr = lambda z: z+1e-9 + epsilon*((z**2).mean()**.5)
        
      w = rho(w)  
      # b = rho(b)                              #bias ignored 'cause we want relevane to only flow to the input neuron and not end up in static bias neurons
        
      z = incr(activations[layer].dot(w))       # step 1 : this step determines the sum of influences for each neuron in the higher layer, analogous to a modified forward pass
      s = relevances[layer+1]/z                 # step 2 : element-wise division as per formula in the paper
      c = s.dot(w.T)                            # step 3
      # print('activation[layer]: ', activations[layer])
      # print()
      # print('s: ', s)
      # print()
      # print('z: ', z)


      
      relevances[layer] = activations[layer]*c  # step 4 : element-wise product and store the value as relevance of current layer
      # print('layer: ', layers[layer])
      # print('relevances[layer]: ', relevances[layer])
    
    
    # print('relevances[0][0]: ', relevances[0][0])
    #distribut4 768-size relevance of 1st layer to word-level relevance
    bert_embedding_level_relevances = np.zeros((dataset_parameter['max_seq_len'], bert_model_parameter['second_last_layer_size']))

    bert_embeddings = word_level_emb.reshape(dataset_parameter['max_seq_len'], bert_model_parameter['second_last_layer_size']).to('cpu')  #50*768

    if(bert_model_parameter['word_level_mean_way']==1):
      for col in range(bert_model_parameter['second_last_layer_size']):
        bert_embedding_level_relevances[:, col] = (bert_embeddings[:, col]/bert_embeddings[:, col].sum())*relevances[0][0][col]

    elif(bert_model_parameter['word_level_mean_way']==2):
      for row in range(dataset_parameter['max_seq_len']):
        word_level_emb_vertical_weights=[]
        with torch.no_grad():
          word_level_emb_vertical_weights = model.word_level_emb_vertical_weights.cpu().detach().numpy()

        bert_embeddings[row] = bert_embeddings[row]*word_level_emb_vertical_weights
      
      for col in range(bert_model_parameter['second_last_layer_size']):
        # print('bert_embeddings[:, col]: ', bert_embeddings[:, col])
        # print('bert_embeddings[:, col].sum(): ', bert_embeddings[:, col].sum())
        # print('relevances[0][0][col]: ', relevances[0][0][col])

        bert_embedding_level_relevances[:, col] = (bert_embeddings[:, col]/bert_embeddings[:, col].sum())*relevances[0][0][col]

    elif(bert_model_parameter['word_level_mean_way']==3):
      word_level_emb_horizontal_weights = model.word_level_emb_horizontal_weights.cpu().detach().numpy()

      for col in range(bert_model_parameter['second_last_layer_size']):
        bert_embeddings[:, col] = bert_embeddings[:, col]*word_level_emb_horizontal_weights

      for col in range(bert_model_parameter['second_last_layer_size']):
        bert_embedding_level_relevances[:, col] = (bert_embeddings[:, col]/bert_embeddings[:, col].sum())*relevances[0][0][col]

    elif(bert_model_parameter['word_level_mean_way']==4):
      word_level_emb_batch_weights = model.word_level_emb_batch_weights.detach().numpy()
      bert_embeddings = bert_embeddings*word_level_emb_batch_weights
      for col in range(bert_model_parameter['second_last_layer_size']):
        bert_embedding_level_relevances[:, col] = (bert_embeddings[:, col]/bert_embeddings[:, col].sum())*relevances[0][0][col]

    # elif(bert_model_parameter['word_level_mean_way']==5):
    #     bert_embedding_level_relevances[:, col] = (bert_embeddings[:, col]/bert_embeddings[:, col].sum())*relevances[0][0][col]


    input_words_relevances = []
    # print('bert_embedding_level_relevance: ', bert_embedding_level_relevances)

    for word in range(dataset_parameter['max_seq_len']):
      input_words_relevances.append(sum(bert_embedding_level_relevances[word]))




    p = torch.sigmoid(torch.tensor(max_activation)).item()                          #predicted output for the given input  


    #taking softmax of relevance values
    # input_words_relevance = [relevance.item() for relevance in torch.softmax(torch.tensor(word_relevances), dim=0)]
    # input_words_relevance = [val.item() for val in torch.softmax(torch.tensor(input_words_relevance), dim=0)]      
    return (sentence, p, input_words_relevances, output_proba)
      # return (p, input_words_relevance)

import matplotlib
import seaborn as sns
import colorsys

def visualize_word_level_relevance(sentence, word_relevances , only_sentence_word_lrp):
    '''This function visualizes the relevance of each word in input sentence
     params: 
              input sentence: string 
              word_relevances (returned by LRP() function) : array of wo relevances
              only_sentence_word_lrp: boolean, if it is True then show relevance of words present in sentence else relevance of each token upto max_length including [CLS], [PAD] token's relevance
      returns:
           df.style.background_gradient(axis=1, gmap=df.iloc[0], cmap='inferno')
    '''
    relevances = word_relevances


    tokens = sentence.split(' ')    
    if only_sentence_word_lrp:
      relevances = relevances[:len(tokens)]



      df = pd.DataFrame(relevances).T
      df.columns = ['w'+str(i)+"_"+tokens[i] for i in range(len(tokens))]
      return df.style.background_gradient(axis=1, gmap=df.iloc[0], cmap='inferno')
    else:
      #get relevance value of padding tokens as well
      while(len(tokens)!=dataset_parameter['max_seq_len']):
        tokens.append('PAD')
      relevances = relevances[:len(tokens)]


      df = pd.DataFrame(relevances).T
      df.columns = ['w'+str(i)+"_"+tokens[i] for i in range(len(tokens))]
      return df.style.background_gradient(axis=1, gmap=df.iloc[0], cmap='inferno')

def visualize_phrase_level_relevance(sentence, word_relevances):
    '''This function visualizes the relevance of each phrase in input sentence after doing phrase level splitting
     params: 
              input sentence: string 
              word_relevances (returned by LRP() function) : array of wo relevances
              only_sentence_word_lrp: boolean
      returns:
           df.style.background_gradient(axis=1, gmap=df.iloc[0], cmap='inferno')
    '''
    relevances = word_relevances
    tokens = sentence.split(' ')
    relevances = word_relevances[:len(tokens)]
    temp_sentence = '\t'.join(tokens)

    #finding phrases in the input sentences and simultaneously calculating phrase-level relevances
    ctags = predict_with_model(chunk_model, temp_sentence, chunk_tokenizer)  #this function is imported from chunking_model.py


    words_arr = temp_sentence.split('\t')
    phrase_tokens=[]
    phrase_relevances = []

    prev_token= words_arr[0]
    prev_relevance = relevances[0]
    for i in range(1, len(ctags)):

      if ctags[i][0]=='B':
        phrase_tokens.append(prev_token)
        phrase_relevances.append(prev_relevance)

        prev_token=words_arr[i]
        prev_relevance = relevances[i]
      elif ctags[i][0]=='I':
        prev_token= prev_token+'\t'+words_arr[i]
        prev_relevance+=relevances[i]
    phrase_tokens.append(prev_token)
    phrase_relevances.append(prev_relevance)

    #phrase_relevances = my_softmax(phrase_relevances) #taking softmax
    relevance_sum = sum(phrase_relevances)
    relevances_normalized = [r/relevance_sum for r in phrase_relevances]




    df = pd.DataFrame(relevances_normalized).T
    df.columns = ['phrase'+str(i)+"_"+phrase_tokens[i] for i in range(len(phrase_tokens))]
    return df.style.background_gradient(axis=1, gmap=df.iloc[0], cmap='inferno')
