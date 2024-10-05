from tensorflow.keras.layers import LSTM, RNN, GRU

# Window size or the sequence length
N_STEPS = 7
# Lookup step, 1 is the next day
LOOKUP_STEP = 1

# whether to scale feature columns & output price as well
SCALE = True
# whether to shuffle the dataset
SHUFFLE = True
# whether to split the training/testing set by date
SPLIT_BY_DATE = False
# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ["Close", "Open", "High", "Low", "Volume"]

### model parameters

N_LAYERS = 2
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = False

### training parameters

# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "mae"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 20