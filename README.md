# RNN-Decoder-Correlated-Noise
Fistly, thanks for the source code from yihanjiang/Sequential-RNN-Decoder. Our source code is completed based on the yihanjiang. 

## Dependency
- Python (2.7.10+)
- numpy (1.14.1)
- Keras (2.0)
- colorednoise (1.0)
- scikit-commpy (0.3.0) For Commpy, we use a modified version of the original commpy, which is in the folder with name commpy. You don't need to install commpy via pip. The original commpy has a few bugs which is fixed in our version.
- h5py (2.7.0)
- tensorflow (1.5)

###Traditional viterbi decoder performance


$ python conv_codes_benchmark.py -num_block 100 -block_len 100 -snr_test_start -1 -snr_test_end 8 -snr_points 10 -decoding_type hard -channel awgn

$ python conv_codes_benchmark.py -num_block 100 -block_len 100 -snr_test_start -1 -snr_test_end 8 -snr_points 10 -decoding_type unquantized -noise_type awgn

####RNN_decoder performance under standard correlation noise model

$python conv_decoder_relationchannel_cudnn.py -block_len 200 -num_block 100000 -code_rate 2 -test_ratio 2 -batch_size 200 -num_Dec_layer 2 -num_Dec_unit 200  -num_epoch 30 -relation_val 0.8 -train_channel_high 0.0
    

#####RNN_decoder performance under pink noise

python conv_decoder_pnoisechannel_cudnn_newgenerator.py -block_len 100 -num_block 50000 -test_ratio 2 -batch_size 100 -test_batch_size 100 -num_Dec_unit 200 -num_epoch 25 -train_channel_high 0.0