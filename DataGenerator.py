import numpy as np
import tensorflow


class DataGenerator(tensorflow.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, data, batch_size=32, shuffle=True):
        """Initialization"""
        # {id-1: (encoder_inputs[0], decoder_inputs[0], decoder_targets[0]), ...}
        self.data = data
        self.list_IDs = list(data.keys())
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = []
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        encoder_inputs, decoder_inputs, decoder_targets = self.__data_generation(list_IDs_temp)
        # print(encoder_inputs)
        return [encoder_inputs, decoder_inputs], decoder_targets

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        batch_encoder_inputs = []
        batch_decoder_inputs = []
        batch_decoder_targets = []
        for i, ID in enumerate(list_IDs_temp):
            batch_encoder_inputs.append(self.data[ID][0])
            batch_decoder_inputs.append(self.data[ID][1])
            batch_decoder_targets.append(self.data[ID][2])

        return np.array(batch_encoder_inputs), np.array(batch_decoder_inputs), np.array(batch_decoder_targets)

    def get_item(self, index):
        return self.__getitem__(index)