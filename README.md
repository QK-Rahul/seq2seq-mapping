# Example code: Sequence to Sequence mapping using encoder-decoder RNNs using **dpnn** and  **rnn** libraries

* adapted from encoder-decoder-coupling example from [Element-research/rnn/examples] and [char-rnn] by Karpathy

* This example includes -
    * multiple LSTM layered encoder-decoder
    * dropout between stacked LSTM layers
    * input sequences can be of any length
        * I'm not aware of effects of arbitrary length sequences during training for real world tasks
        * inside a batch, all the sequences should be of the same length or you'll get an exception
        * to form batch from variable length sequences, use padding
            * recommended padding style is: {0,0,0,GO,1,2,3,4} for encoder and {1,2,3,4,EOS,0,0,0} for the decoder
	* validation, early-stopping
	* using RMSProp, can easily change to another optimization procedure supported by optim package e.g. adam/adagrad for training
	* saving model at predefined checkpoints and resuming training from saved model
	* running on nvidia GPU
	* two Synthetic data sets

* NOTE on using a saved model
    * If you run your experiment on GPU then before using the saved model, convert it to a cpu model first using convert_gpuCheckpoint_to_cpu.lua

* Choose from two tasks

        th seq2seq.lua -synthetic 1
        th seq2seq.lua -synthetic 2

* TODO -
    * sampling from saved model
    * removing teacher forcing


[Element-research/rnn/examples]: <https://github.com/Element-Research/rnn/blob/master/examples/encoder-decoder-coupling.lua>
[char-rnn]: <https://github.com/karpathy/char-rnn>
