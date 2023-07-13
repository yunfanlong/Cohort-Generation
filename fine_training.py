#!/usr/bin/env python
# coding: utf-8

# # Semi-Supervised Model Training

# Part 1: Unsupervised NLP - Encoder Training

# Import Packages (fastai for optimizing inference)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
from fastai.text import *
import pandas as pd
from fastai.callbacks.tracker import *
from keras_tqdm import TQDMNotebookCallback
from tqdm.auto import tqdm
tqdm.pandas()
from fastai.distributed import *
from utils.model import *
from utils.helper import *

if __name__ == "__main__":
    #list of items to remove
    to_remove = ["\d+", "\r", "\n","IMPRESSION:"]

    # file path
    text_file_path = '../data/reports_all.csv'

    # load in data
    all_text_data = pd.read_csv(text_file_path,index_col=0);all_text_data.head()

    # clean data by removing items from reports
    cleaned_all_text_data = clean_impressions(all_text_data,'report',to_remove)

    # development set split
    train_set, val_set, test_set = shuffle_split_dataset(cleaned_all_text_data)

    # convert raw data into language model understandable format
    text_data = report_slicer(TextLMDataBunch,"../data/",
                                    train_df=train_set,
                                    valid_df=val_set,
                                    test_df=test_set,
                                    text_cols='report')


    # Saving and Loading Data
    # save model
    text_data.save('../data/reports_all.pkl')

    # load model
    text_data = load_data("../data/", 'reports_all.pkl')


    # Basic Encoder Model Parameters Set Here
    vocab_size = len(text_data.vocab.itos)
    encoding_size = 400
    layer_size = 1152
    num_layers = 3


    #stable, these variables don't need to be fine-tuned (described down below)
    pad_token = 1
    hidden_dropout = 0.2
    input_dropout = 0.6
    embed_dropout = 0.1
    weight_dropout = 0.5
    output_dropout = 0.1
    qrnn_cells = False
    bidirectional = False


    # Build Encoder 
    encoder = us_ccds_encoder(vocab_sz=vocab_size,
                            enc_sz=encoding_size, 
                            n_hid=layer_size, 
                            n_layers=num_layers, 
                            pad_token=pad_token, 
                            hidden_p=hidden_dropout,
                            input_p=input_dropout, 
                            embed_p=embed_dropout, 
                            weight_p=weight_dropout, 
                            qrnn=qrnn_cells, 
                            bidir=bidirectional)

    # Build Decoder
    decoder = us_ccds_decoder(n_out=vocab_size, 
                            n_hid=encoding_size, 
                            output_p=output_dropout, 
                            tie_encoder=encoder.encoder, 
                            bias=True)



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
    # - **cyc_len** = epochs/iterations
    # - **max_lr** = learning rate
    # - **callbacks** = saved model with lowest validation loss during training session as 'bestmodel'
    # - **to load best model parameters:** `{model_name}.load('bestmodel')`
    learn_awd.fit_one_cycle(cyc_len=20,max_lr=1e-3,callbacks=[SaveModelCallback(learn_awd, every='improvement', monitor='valid_loss')])

    # Save/load parameters
    learn_awd.save('language_modelv1')
    learn_awd.load('language_modelv1')

    # Run predict on series of text to explore model language understanding
    learn_awd.predict("No evidence of a particular",100)
    learn_awd.model[0].hidden_dps

    # Part 2: Supervised NLP - Decoder Training

    # Load and clean data [using same methods](#proc)
    data_file_path = '../data/abdoAA_500.csv'
    all_labeled_data = pd.read_csv(data_file_path);all_labeled_data.head()
    cleaned_all_labeled_data = clean_impressions(all_labeled_data,'IMPRESSION',to_remove)
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
    data_labeled.save('../data/AAA.pkl')
    data_labeled = load_data('../data','AAA.pkl')

    # Basic Decoder Model Parameters Set Here
    num_classes = 2
    decoder_layer_sizes =[50]
    decoder_dropout =[0.1]
    bptt_classifier = 70
    max_length = 2000

    # Setup for Supervised Models
    pretrained = learn_awd.model[0]
    encoder = su_ccds_encoder(bptt=bptt_classifier, 
                                max_len=max_length,
                                custom_model=pretrained,
                                pad_idx=pad_token)

    decoder = su_ccds_decoder(num_classes,
                            encoding_sz=encoding_size,
                            lin_ftrs=decoder_layer_sizes,
                            ps=decoder_dropout)


    # `text_classifier` is the function that creates the combined supervised model
    learn = text_classifier(data_labeled, encoder, decoder)

    # Freeze the encoder model (`learn.unfreeze()` allows for fine-tuning of encoder model, too)
    learn.freeze()
    learn.model
    learn_awd.lr_find()
    learn_awd.recorder.plot()

    # Start Training Process
    learn.fit_one_cycle(cyc_len=50,max_lr=1e-3,callbacks=[SaveModelCallback(learn_awd, every='improvement', monitor='valid_loss')])
    learn.save('AAA')

    # Predict on string easily with `{model_name}.predict({string})`
    learn.predict('Pizza time')
    learn.predict('Stable postsurgical appearance of graft replacement of the ascending aorta, arch, and descending thoracic aorta. Minimal decrease in size of fluid collection surrounding the descending thoracic graft and adjacent small pleural effusion.')
    learn.predict('No evidence of abdominal aortic aneurysm.')

    # Good command if GPU util blows up
    torch.cuda.empty_cache()