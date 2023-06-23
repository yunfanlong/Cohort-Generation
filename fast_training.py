# Semi-Supervised Model Training

# Part 1: Unsupervised NLP - Encoder Training

# Import Packages (fastai for optimizing inference)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from fastai.text import *
import pandas as pd
from fastai.callbacks.tracker import *
from keras_tqdm import TQDMNotebookCallback
from tqdm.auto import tqdm
tqdm.pandas()
from fastai.distributed import *
from functools import wraps


# shuffle_split_dataset takes in five inputs:
# - data = pandas dataframe
# - train/val/test_frac = ratio for each development set
# - seed = random seed for mantain consistency in testing

def shuffle_split_dataset(data,train_frac=.85,val_frac=.1,test_frac=.05,seed=42):
    
    data = data.sample(frac=1,random_state=seed)
    train_index = int(data.count()[0]*train_frac)
    val_index = train_index + int(data.count()[0]*val_frac)
    
    train_set = data[:train_index]
    val_set = data[train_index:val_index]
    test_set = data[val_index:]
    return train_set, val_set, test_set

def add_method(cls):
    def decorator(func):
        @wraps(func) 
        def wrapper(self, *args, kwargs): 
            return func(*args, kwargs)
        setattr(cls, func.__name__, wrapper)
        return func 
    return decorator

# Tokenization - _get_processor
# 
# - xxunk = unknown word 
# - xxbos = beginning of a report 
# - xxfld = separate columns of text 
# - xxmaj = uppercase letter 
# - xxup = all caps 
# - xxrep n {char} = next character is repeated n times 
# - xxwrep n {word} = next word is repeated n times
# 
# The tokenization strategy is set to [scapy](https://spacy.io/api/), but a custom version can be inputted into method. This method is input into the [report_slicer method](#report_slicer)
def _get_processor(tokenizer:Tokenizer=None, vocab:Vocab=None, chunksize:int=10000, max_vocab:int=60000,
                   min_freq:int=2, mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False):
    return [TokenizeProcessor(tokenizer=tokenizer, chunksize=chunksize, 
                              mark_fields=mark_fields, include_bos=include_bos, include_eos=include_eos),
            NumericalizeProcessor(vocab=vocab, max_vocab=max_vocab, min_freq=min_freq)]

# report_slicer processes the raw text data into model-readable form
# [TextLMDataBunch](https://docs.fast.ai/text.data.html) is a fastai class that manages data for development, and makes it a lot easier to train. Report Slicer uses the class to store the data so that the language model can actually take in individual vocab tokens as inputs. Report Slicer's inputs are:
# 
# - cls = class (always TextLMDataBunch)
# - path = directory where data is stored and where saved model files will be saved and loaded
# - train/valid/test_df = development sets from [shuffle_split_dataset method](#shuffle_split_dataset) 
# - text_cols = the header of the column with text (can be a list of multiple columns of text if necessary, in which case, mark_fields should be set to <font color=blue>True</font>
# - tokenizer = tokenizer function (set in [_get_processor](#token))
# - vocab = not required, list of vocab to corresponding, method will automatically create initially
# - chunksize = iterator that loads portions of dataframe for processing, given # of characters per 'chunk'
# - max_vocab = max vocab in dataset
# - mark_fields = token to mark seperation between more than one text column for training
@add_method(TextLMDataBunch)
def report_slicer(cls, path:PathOrStr, train_df:DataFrame, valid_df:DataFrame, test_df:Optional[DataFrame]=None,
                tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, text_cols:IntsOrStrs=1,
                label_cols:IntsOrStrs=0, label_delim:str=None, chunksize:int=10000, max_vocab:int=60000,
                min_freq:int=2, mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False, kwargs) -> DataBunch:
    
        processor = _get_processor(tokenizer=tokenizer, vocab=vocab, chunksize=chunksize, max_vocab=max_vocab,
                                   min_freq=min_freq, mark_fields=mark_fields, 
                                   include_bos=include_bos, include_eos=include_eos)
        if classes is None and is_listy(label_cols) and len(label_cols) > 1: classes = label_cols
        src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),
                        TextList.from_df(valid_df, path, cols=text_cols, processor=processor))
        if cls==TextLMDataBunch: src = src.label_for_lm()
        else: 
            if label_delim is not None: src = src.label_from_df(cols=label_cols, classes=classes, label_delim=label_delim)
            else: src = src.label_from_df(cols=label_cols, classes=classes)
        if test_df is not None: src.add_test(TextList.from_df(test_df, path, cols=text_cols))
        return src.databunch(kwargs)

text_data = report_slicer(TextLMDataBunch,"../data/",
                                  train_df=train_set,
                                  valid_df=val_set,
                                  test_df=test_set,
                                  text_cols='report')

# Best unsupervised test loss:
# - Unidirectional = 1.64
# - Bidirectional = 1.14 (loss was consistently decreasing, not converged but didn't have enough time)

# Layer Modules - Setup for Unsupervised Models for Training
# 
# - Dropout Mask - Form of inverted-dropout, to make testing easier, based on concept [here](https://deepnotes.io/dropout)
# - RNNDropout - layer with Dropout Mask
# - Weight/Embedding Dropout - Dropout classes for respective layers (can be 'turned off' but best results if enabled), otherwise it overfits like crazy
# - "Split" functions allow for gradual fine-tuning of language model and classifier by making the learning rate tiny in the early layers and bigger in the later ones. This idea is from [this paper](https://arxiv.org/pdf/1711.10177.pdf) and is best used for the classifier training, especially on reports with super fine-grained classes, actually does not even take as long
# - the config modules are taken from the [AWD-LSTM paper](https://arxiv.org/pdf/1708.02182.pdf), they're needed if wanna utilize the pretrained models, from WikiText data
# - [link to split function](https://github.com/fastai/fastai/blob/master/fastai/text/models/awd_lstm.py)
def dropout_mask(x:Tensor, sz:Collection[int], p:float):
    return x.new(*sz).bernoulli_(1-p).div_(1-p)

class RNNDropout(Module):
    def __init__(self, p:float=0.5): self.p=p

    def forward(self, x:Tensor)->Tensor:
        if not self.training or self.p == 0.: return x
        m = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x * m

class WeightDropout(Module):
    def __init__(self, module:nn.Module, weight_p:float, layer_names:Collection[str]=['weight_hh_l0']):
        self.module,self.weight_p,self.layer_names = module,weight_p,layer_names
        for layer in self.layer_names:
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(w, p=self.weight_p, training=False)

    def _setweights(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=self.training)

    def forward(self, *args:ArgStar):
        self._setweights()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=False)
        if hasattr(self.module, 'reset'): self.module.reset()

class EmbeddingDropout(Module):
    #Dropout

    def __init__(self, emb:nn.Module, embed_p:float):
        self.emb,self.embed_p = emb,embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None: self.pad_idx = -1

    def forward(self, words:LongTensor, scale:Optional[float]=None)->Tensor:
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0),1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else: masked_embed = self.emb.weight
        if scale: masked_embed.mul_(scale)
        return F.embedding(words, masked_embed, self.pad_idx, self.emb.max_norm,
                           self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)

class SequentialRNN(nn.Sequential):
    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'): c.reset()

def awd_lstm_lm_split(model:nn.Module) -> List[nn.Module]:
    groups = [[rnn, dp] for rnn, dp in zip(model[0].rnns, model[0].hidden_dps)]
    return groups + [[model[0].encoder, model[0].encoder_dp, model[1]]]

def awd_lstm_clas_split(model:nn.Module) -> List[nn.Module]:
    groups = [[model[0].module.encoder, model[0].module.encoder_dp]]
    groups += [[rnn, dp] for rnn, dp in zip(model[0].module.rnns, model[0].module.hidden_dps)]
    return groups + [[model[1]]]


# Setup for Unsupervised Models: Encoder + Decoder
# Encoder
# [Setup Here](build_encoder)
# - The encoder can be modified as needed, but the us_ccds_encoder class can take in various modifiers pretty nicely
# - The hyperparameters that are currently the default were the ones that worked well, specifically on the CT Hemorrhage dataset
# - vocab_sz = should not be modified, is automatically set to the vocab size of the processed dataset
# - emb_sz = size of embedding later
# - n_hid = dimensionality of hidden layers
# - n_layers = number of layers
# - hidden_p = dropout % in hidden layers
# - input_p = dropout % in input layers
# - embed_p = drop % in embedding layers
# - weight_p = drop % for weights
# - qrnn = quasi-recurrent cells [based on this paper](https://arxiv.org/pdf/1611.01576.pdf) - supposedly it's a better technique, but it's terrible when I use it, so either something wrong with the way I did it or its just bad
# - bidir - Bidirectional model - got it working now, so it's a thing to try, actually seems to be doing better than the unidirectional one, seems to take longer to train (+10m/epoch)
class us_ccds_encoder(Module):
    #Encoder
    initrange=0.1

    def __init__(self, vocab_sz:int=vocab_size, enc_sz:int=encoding_size, n_hid:int=layer_size, n_layers:int=num_layers, pad_token:int=pad_token, hidden_p:float=hidden_dropout,
                 input_p:float=input_dropout, embed_p:float=embed_dropout, weight_p:float=weight_dropout, qrnn:bool=qrnn_cells, bidir:bool=bidirectional):
        self.bs,self.qrnn,self.enc_sz,self.n_hid,self.n_layers = 1,qrnn,enc_sz,n_hid,n_layers
        self.n_dir = 2 if bidir else 1
        self.encoder = nn.Embedding(vocab_sz, enc_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        if self.qrnn:
            from .qrnn import QRNN
            self.rnns = [QRNN(enc_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else enc_sz)//self.n_dir, 1,
                              save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True, bidirectional=bidir) 
                         for l in range(n_layers)]
            for rnn in self.rnns: 
                rnn.layers[0].linear = WeightDropout(rnn.layers[0].linear, weight_p, layer_names=['weight'])
        else:
            self.rnns = [nn.LSTM(enc_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else enc_sz)//self.n_dir, 1,
                                 batch_first=True, bidirectional=bidir) for l in range(n_layers)]
            self.rnns = [WeightDropout(rnn, weight_p) for rnn in self.rnns]
        self.rnns = nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])

    def forward(self, input:Tensor, from_embeddings:bool=False)->Tuple[Tensor,Tensor]:
        if from_embeddings: bs,sl,es = input.size()
        else: bs,sl = input.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()
        raw_output = self.input_dp(input if from_embeddings else self.encoder_dp(input))
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
        self.hidden = to_detach(new_hidden, cpu=False)
        return raw_outputs, outputs

    def _one_hidden(self, l:int)->Tensor:
        nh = (self.n_hid if l != self.n_layers - 1 else self.enc_sz) // self.n_dir
        return one_param(self).new(self.n_dir, self.bs, nh).zero_()

    def select_hidden(self, idxs):
        if self.qrnn: self.hidden = [h[:,idxs,:] for h in self.hidden]
        else: self.hidden = [(h[0][:,idxs,:],h[1][:,idxs,:]) for h in self.hidden]
        self.bs = len(idxs)

    def reset(self):
        [r.reset() for r in self.rnns if hasattr(r, 'reset')]
        if self.qrnn: self.hidden = [self._one_hidden(l) for l in range(self.n_layers)]
        else: self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]

# Decoder
# [Setup Here](#build_decoder)
# - Decoder's simple, just a fully connected layer, out to the vocabulary layer
# - n_out - output size of decoder, should remain the same as the embedding size for encoder, which is the same as the vocab size of the processed dataset
# - n_hid - size of encoding layer from encoder, automatically set
# - output_p = % dropout for last layer
# - tie_encoder - important input to connect encoder weights to decoder weights, otherwise gradients aren't computed
# - bias - bias for last layer
class us_ccds_decoder(Module):
    #Decoder
    initrange=0.1

    def __init__(self, n_out:int=vocab_size, n_hid:int=encoding_size, output_p:float=output_dropout, tie_encoder:nn.Module=None, bias:bool=True):
        self.decoder = nn.Linear(n_hid, n_out, bias=bias)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.output_dp = RNNDropout(output_p)
        if bias: self.decoder.bias.data.zero_()
        if tie_encoder: self.decoder.weight = tie_encoder.weight

    def forward(self, input:Tuple[Tensor,Tensor])->Tuple[Tensor,Tensor,Tensor]:
        raw_outputs, outputs = input
        output = self.output_dp(outputs[-1])
        decoded = self.decoder(output)
        return decoded, raw_outputs, outputs


# load_pretrained - method to load in pretrained models
# - from online article, loads in pretrained files and transfers weights, still not fully functional | will try and get working by Friday

def load_pretrained(self, wgts_fname:str, itos_fname:str, strict:bool=True):
        old_itos = pickle.load(open(itos_fname, 'rb'))
        old_stoi = {v:k for k,v in enumerate(old_itos)}
        wgts = torch.load(wgts_fname, map_location=lambda storage, loc: storage)
        if 'model' in wgts: wgts = wgts['model']
        wgts = convert_weights(wgts, old_stoi, self.data.train_ds.vocab.itos)
        self.model.load_state_dict(wgts, strict=strict)
        return self

# Useful method to save only encoder section of language model

def save_encoder(self, name):
        if is_pathlike(name): 
            self._test_writeable_path()
        encoder = get_model(self.model)[0]
        if hasattr(encoder, 'module'): 
            encoder = encoder.module
        torch.save(encoder.state_dict(), self.path/self.model_dir/f'{name}.pth')


# create_labeled_set processes the raw text data into model-readable form
# [TextLMDataBunch](https://docs.fast.ai/text.data.html) is a fastai class that manages data for development, and makes it a lot easier to train. It inputs are:
# - cls = class (always TextClasDataBunch)
# - path = directory where data is stored and where saved model files will be saved and loaded
# - train/valid/test_df = development sets from [shuffle_split_dataset method](#shuffle_split_dataset) 
# - text_cols = the header of the column with text (can be a list of multiple columns of text if necessary, in which case, mark_fields should be set to <font color=blue>True</font>
# - label_cols = the header of the column with labels (can be a list of multiple columns of labels if necessary, results in multilabel or multiclass classification depending on dataset.)
# - tokenizer = tokenizer function (set in [_get_processor](#token))
# - bs = batch_size of function, usually 128 is the limit (for 32 GB of GPUs)
# - vocab = will automatically create initially, required to be set to the same vocab as the previous dataframe, generated by the report_slicer method 
# - if there is an error while training, relating to the dataframe, make sure that the processing variables not above are set to the same values as the encoder dataframe object
@add_method(TextClasDataBunch)
def create_labeled_set(cls, path:PathOrStr, train_df:DataFrame, valid_df:DataFrame, test_df:Optional[DataFrame]=None,
            tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, text_cols:IntsOrStrs=1,
            label_cols:IntsOrStrs=0, label_delim:str=None, chunksize:int=10000, max_vocab:int=60000,
            min_freq:int=2, mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False, kwargs) -> DataBunch:
    processor = _get_processor(tokenizer=tokenizer, vocab=vocab, chunksize=chunksize, max_vocab=max_vocab,
                               min_freq=min_freq, mark_fields=mark_fields, 
                               include_bos=include_bos, include_eos=include_eos)
    if classes is None and is_listy(label_cols) and len(label_cols) > 1: 
        classes = label_cols
    src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),
                    TextList.from_df(valid_df, path, cols=text_cols, processor=processor))
    if cls==TextLMDataBunch: 
        src = src.label_for_lm()
    else: 
        if label_delim is not None: 
            src = src.label_from_df(cols=label_cols, classes=classes, label_delim=label_delim)
        else: src = src.label_from_df(cols=label_cols, classes=classes)
    if test_df is not None: 
        src.add_test(TextList.from_df(test_df, path, cols=text_cols))
    return src.databunch(kwargs)


data_labeled = create_labeled_set(TextClasDataBunch,
                             path='../data/',
                             train_df=labeled_train_set,
                             valid_df=labeled_val_set,
                             test_df=labeled_test_set,
                             text_cols='IMPRESSION',
                             label_cols='cohort',
                             vocab=text_data.train_ds.vocab,
                             bs=128
                             )


# Best Classifier Results (Test Set)
# Binary:
# - CT Hemorrhage = AUC:0.99
# - Occlusion = AUC: 0.98
# - Stroke = AUC: 0.95
# 
# Multiclass:
# - CT Hemorrhage:
#     - EDH = RMSE: 1.52
#     - IPH = RMSE: 0.84
#     - IVH = RMSE: 1.37
#     - SAH = RMSE: 1.65
#     - SDH = RMSE: 2.02
#     
# - Occlusion:
#     - M1 = AUC: 0.95
#     - M2 = AUC: 0.93
#     - ICA = AUC: 0.92
#     - Basilar = AUC: 0.84

# masked_concat_pool:
# - based on [this paper](https://arxiv.org/pdf/1406.4729.pdf)
# - strategy developed for usage on ConvNets, but increased the performance of this model quite a lot (+0.1 AUC)
# - concatenates max and average pools - adaptive max pooling
def masked_concat_pool(outputs, mask):
    output = outputs[-1]
    avg_pool = output.masked_fill(mask[:, :, None], 0).mean(dim=1)
    avg_pool *= output.size(1) / (output.size(1)-mask.type(avg_pool.dtype).sum(dim=1))[:,None]
    max_pool = output.masked_fill(mask[:,:,None], -float('inf')).max(dim=1)[0]
    x = torch.cat([output[:,-1], max_pool, avg_pool], 1)
    return x

# Setup for Supervised Models

# Encoder
# - MultiBatchEncoder transforms the standard report encoder model trained in the previous section by wrapping it to process variable length strings and export a singular encoding, which is then tossed into the decoder model
# - bptt = backprop over time, samples before gradients are calculated
# - max_len = maximum length of text data input, best if kept large, but biggest constraints are memory related + maybe want to restrict sentence size
# - module = preset, takes in initial model to convert to MultiBatchEncoder, set to encoder model
class MultiBatchEncoder(Module):
    def __init__(self, bptt:int, max_len:int, module:nn.Module, pad_idx:int=1):
        self.max_len,self.bptt,self.module,self.pad_idx = max_len,bptt,module,pad_idx

    def concat(self, arrs:Collection[Tensor])->Tensor:
        return [torch.cat([l[si] for l in arrs], dim=1) for si in range_of(arrs[0])]

    def reset(self):
        if hasattr(self.module, 'reset'): self.module.reset()

    def forward(self, input:LongTensor)->Tuple[Tensor,Tensor]:
        bs,sl = input.size()
        self.reset()
        raw_outputs,outputs,masks = [],[],[]
        for i in range(0, sl, self.bptt):
            r, o = self.module(input[:,i: min(i+self.bptt, sl)])
            if i>(sl-self.max_len):
                masks.append(input[:,i: min(i+self.bptt, sl)] == self.pad_idx)
                raw_outputs.append(r)
                outputs.append(o)
        return self.concat(raw_outputs),self.concat(outputs),torch.cat(masks,dim=1)


#  us_ccds_encoder builds decoder
# - bptt = backprop through time, number of samples before gradients calculated
# - max_len = max length of sequence of text
# - *pretrained* = *still in progress,* will allow for loading of pretrained external model on PubMed
# - custom_model = name of custom model object
def su_ccds_encoder(bptt:int=bptt_classifier, 
                    max_len:int=max_length,
                    custom_model=None,
                    pad_idx=pad_token) -> nn.Module:
    
    encoder = MultiBatchEncoder(bptt, max_len, custom_model.cuda(), pad_idx=pad_idx)
    return encoder

# us_ccds_decoder builds decoder
# - vocab_sz = vocab size for dataset, leave unchanged
# - n_class = number of classes, automaticaaly set from variable
# - lin_ftrs = list of size of layers in decoder, can be variable length to create variable depth decoders
# - output_p = list of dropout % that is equal to len(lin_ftrs) otherwise set all to 0.1
def su_ccds_decoder(n_class:int=num_classes,
                    lin_ftrs:Collection[int]=decoder_layer_sizes,
                    encoding_sz:int=encoding_size,
                    ps:Collection[float]=decoder_dropout) -> nn.Module:
    if ps is None:  
        ps = [0.1]*len(lin_ftrs)
    layers = [encoding_size * 3] + lin_ftrs + [n_class]
    ps = [0.1] + ps
    decoder = PoolingLinearClassifier(layers, ps)
    return decoder

# text_classifier_learner is the function that creates the combined supervised model
# - data = data_class object that contains processed data
# - encoder = encoder model
# - decoder = decoder model
# 
def text_classifier(data:DataBunch,
                            encoder=encoder,
                            decoder=decoder,
                            learn_kwargs) -> 'TextClassifierLearner':
    model = SequentialRNN(encoder, decoder)
    learn = RNNLearner(data, model, split_func=awd_lstm_clas_split, learn_kwargs)
    return learn

if __name__ == "__main__":
    # Cleaning methods, just add terms to remove in to_remove
    to_remove = ["\d+", "\r", "\n","IMPRESSION:"]
    def clean_impressions(data,column):
        data[column] = data[column].str.strip()
        for item in to_remove:
            data[column] = data[column].str.replace(item, '')
        return data

    text_file_path = '../data/reports_all.csv'

    all_text_data = pd.read_csv(text_file_path,index_col=0);all_text_data.head()

    cleaned_all_text_data = clean_impressions(all_text_data,'report')

    train_set, val_set, test_set = shuffle_split_dataset(cleaned_all_text_data)

    # Helper function to add methods to classes
    from functools import wraps

    def add_method(cls):
        def decorator(func):
            @wraps(func) 
            def wrapper(self, *args, kwargs): 
                return func(*args, kwargs)
            setattr(cls, func.__name__, wrapper)
            return func 
        return decorator
    text_data = report_slicer(TextLMDataBunch,"../data/",
                                  train_df=train_set,
                                  valid_df=val_set,
                                  test_df=test_set,
                                  text_cols='report')


    # Saving and Loading Data
    # - running {data_variable}.save({path_to_saved_file}) will save weights of the model, but not the model structure
    # - running {data_variable} = load_data({path_to_saved_file}) will again load the weights of model if structure is the same
    text_data.save('../data/reports_all.pkl')
    text_data = load_data("../data/", 'reports_all.pkl')


    # Basic Encoder Model Parameters Set Here
    # - already set, vocab size is set to the length of the dataset vocabulary (after processing) 
    # - encoding_size = encoding size, yeah
    # - layer_size = size of hidden layers
    # - num layers = # of hidden layers
    # - rest of parameters (which don't need to be modified) [described here](#us_en)


    vocab_size = len(text_data.vocab.itos)
    encoding_size = 400
    layer_size = 1152
    num_layers = 3

    # stable, these variables don't need to be fine-tuned (described down below)
    pad_token = 1
    hidden_dropout = 0.2
    input_dropout = 0.6
    embed_dropout = 0.1
    weight_dropout = 0.5
    output_dropout = 0.1
    qrnn_cells = False
    bidirectional = True

    # Build Encoder [(Descriptions of arguments here)](#us_en)
    encoder = us_ccds_encoder()

    # Build Decoder ([Descriptions of arguments here](#us_de))
    decoder = us_ccds_decoder()

    # Combine encoder + decoder into Unsupervised-Model
    custom_model = SequentialRNN(encoder,decoder)
    learn_awd = LanguageLearner(text_data, custom_model)


    # Convert to run on GPU
    learn_awd.model = custom_model.cuda()
    learn_awd.to_parallel()
    # Change Loss Function if Needed

    #learn_awd.loss_func=MSELossFlat()

    # Find & Plot Learning Rate
    learn_awd.lr_find()
    learn_awd.recorder.plot()


    # Start Training Process
    # - cyc_len = epochs/iterations
    # - max_lr = learning rate
    # - callbacks = saved model with lowest validation loss during training session as 'bestmodel'
    # - to load best model parameters: {model_name}.load('bestmodel')
    learn_awd.fit_one_cycle(cyc_len=1,max_lr=1e-3,callbacks=[SaveModelCallback(learn_awd, every='improvement', monitor='valid_loss')])

    # Load bestmodel parameters
    learn_awd.save('language_modelv1')
    learn_awd.load('language_modelv1')

    # Run predict on series of text to explore model language understanding
    learn_awd.predict("No evidence of a particular",100)

    learn_awd.model

    # Part 2: Supervised NLP - Decoder Training

    # Load and clean data [using same methods](#proc)
    data_file_path = '../data/abdoAA_500.csv'

    all_labeled_data = pd.read_csv(data_file_path);all_labeled_data.head()

    cleaned_all_labeled_data = clean_impressions(all_labeled_data,'IMPRESSION')

    labeled_train_set, labeled_val_set, labeled_test_set = shuffle_split_dataset(cleaned_all_labeled_data)

    data_labeled = create_labeled_set(TextClasDataBunch,
                             path='../data/',
                             train_df=labeled_train_set,
                             valid_df=labeled_val_set,
                             test_df=labeled_test_set,
                             text_cols='IMPRESSION',
                             label_cols='cohort',
                             vocab=text_data.train_ds.vocab,
                             bs=128
                             )

    # Saving and Loading Data
    # - running {data_variable}.save({path_to_saved_file}) will save weights of the model, but not the model structure
    # - running {data_variable} = load_data({path_to_saved_file}) will again load the weights of model if structure is the same
    data_labeled.save('../data/AAA.pkl')
    data_labeled = load_data('../data','AAA.pkl')

    # Basic Decoder Model Parameters Set Here
    # - encoding_size = encoding size from encoder training
    # - num_classes = number of classes, yeah
    # - decoder_layer_sizes = the sizes of the layers of decoder, can be variable length, just pass any length list of layer sizes
    # - decoder_dropout = the corresponding dropout values for the decoder_layer_sizes
    # - bptt_classifier = backprop through time, number of samples before gradient calculation
    # - max_length = maximum length of text to input into algorithm
    num_classes = 2
    decoder_layer_sizes =[50]
    decoder_dropout = [0.1]
    bptt_classifier = 70
    max_length = 2000

    pretrained = learn_awd.model[0]
    encoder = su_ccds_encoder(custom_model=pretrained)
    decoder = su_ccds_decoder()

    learn = text_classifier(data_labeled, metrics = [AUROC()])

    # Freeze the encoder model (learn.unfreeze() allows for fine-tuning of encoder model, too)
    learn.freeze()
    learn.model
    learn_awd.lr_find()
    learn_awd.recorder.plot()


    # Start Training Process
    # - cyc_len = epochs/iterations
    # - max_lr = learning rate
    # - callbacks = saved model with lowest validation loss during training session as 'bestmodel'
    # - to load best model parameters: {model_name}.load('bestmodel')
    learn.fit_one_cycle(cyc_len=50,max_lr=1e-3,callbacks=[SaveModelCallback(learn_awd, every='improvement', monitor='valid_loss')])
    learn.save('AAA_model')
    learn.load('AAA_model')


    # Predict on string easily with {model_name}.predict({string})
    learn.predict('Pizza time')
    learn.predict('Stable postsurgical appearance of graft replacement of the ascending aorta, arch, and descending thoracic aorta. Minimal decrease in size of fluid collection surrounding the descending thoracic graft and adjacent small pleural effusion.')
    learn.predict('No evidence of abdominal aortic aneurysm.')
    torch.cuda.empty_cache()