import tensorflow as tf
from keras import layers
from tensorflow.keras.layers import Dense, Flatten, Convolution1D, MaxPooling1D, GRU

# Define a custom layer ProbAttention that inherits from the base Layer class
class ProbAttention(layers.Layer):
    def __init__(self, embedding_dim, num_heads, dropout, conv_kernel_size, pool_kernel_size, sample_k, **kwargs):
        # Initialize the layer with key hyperparameters for multi-head attention
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.key_dim = self.embedding_dim // self.num_heads  # Calculate the dimension for each head
        self.dropout = dropout
        self.sample_k = sample_k

        # Initialize dense layers for transforming input into query, key, and value spaces
        self.query_dense = layers.Dense(embedding_dim)
        self.key_dense = layers.Dense(embedding_dim)
        self.value_dense = layers.Dense(embedding_dim)
        
        # Initialize a dropout layer
        self.Dropout = layers.Dropout(dropout)
        
        # Convolution and pooling kernel sizes that dictate the operation
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size

        # Initialize convolution layer for output processing
        self.conv = layers.Conv1D(filters=num_heads * self.key_dim, 
                                  kernel_size=conv_kernel_size, 
                                  padding='same')
        
        # Initialize ELU activation layer
        self.elu = layers.ELU()
        
        # Initialize max pooling layer for down-sampling
        self.max_pool = layers.MaxPooling1D(pool_size=pool_kernel_size, strides=pool_kernel_size)

    def get_config(self):
        # Return the configuration of the layer for saving/loading purposes
        config = super(ProbAttention, self).get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
            'sample_k': self.sample_k,
            'conv_kernel_size': self.conv_kernel_size,
            'pool_kernel_size': self.pool_kernel_size,
        })
        return config

    def build(self, input_shape):
        # Define learnable weights for query, key, and value linear transformations
        self.Wq = self.add_weight(name='1', shape=(input_shape[-1], self.key_dim * self.num_heads),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.Wk = self.add_weight(name='2', shape=(input_shape[-1], self.key_dim * self.num_heads),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.Wv = self.add_weight(name='3', shape=(input_shape[-1], self.key_dim * self.num_heads),
                                  initializer='glorot_uniform',
                                  trainable=True)
        # Combine heads into the original input dimension space
        self.combine_heads = layers.Dense(input_shape[-1])

    def call(self, inputs):
        # Process input through the attention mechanism
        batch_size = tf.shape(inputs)[0]
        Q = inputs
        K = inputs
        V = inputs

        # Perform dense transformations
        Q = tf.matmul(Q, self.Wq)
        K = tf.matmul(K, self.Wk)
        V = tf.matmul(V, self.Wv)

        # Reshape tensors for separate attention heads
        Q = tf.reshape(Q, (tf.shape(Q)[0], tf.shape(Q)[1], self.num_heads, self.key_dim))
        K = tf.reshape(K, (tf.shape(K)[0], tf.shape(K)[1], self.num_heads, self.key_dim))
        V = tf.reshape(V, (tf.shape(V)[0], tf.shape(V)[1], self.num_heads, self.key_dim))
        
        # Transpose to bring head dimension forward for matrix operations
        Q = tf.transpose(Q, [0, 2, 1, 3])
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.transpose(V, [0, 2, 1, 3])

        L = tf.shape(K)[2]

        # Sample from keys and values for probabilistic attention
        sample_indices = tf.random.uniform(shape=(self.sample_k,), maxval=L, dtype=tf.int32)
        K_sampled = tf.gather(K, sample_indices, axis=2)
        V_sampled = tf.gather(V, sample_indices, axis=2)

        # Compute scaled dot-product attention
        attention_scores = tf.matmul(Q, K_sampled, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(float(self.key_dim))
        
        # Gather sampled attention scores
        attention_scores_sampled = tf.gather(attention_scores, sample_indices, axis=3)

        # Apply softmax to normalize scores
        attention_weights = tf.nn.softmax(attention_scores_sampled, axis=3)
        
        # Compute output using weighted sum of values
        output = tf.matmul(attention_weights, V_sampled)

        # Transpose and reshape tensor to combine head representations
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.num_heads * self.key_dim))
        
        # Pass through convolution layer
        output = self.conv(output)

        # Apply ELU activation
        output = self.elu(output)

        # Down-sample through max pooling
        output = self.max_pool(output)

        # Combine and apply dropout
        output = self.combine_heads(output)
        output = self.Dropout(output)    
        return output

# Define a simple CNN model with a few convolutional and pooling layers
class CNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # First layer: Convolution1D followed by max pooling
        self.conv1 = Convolution1D(filters=32, kernel_size=3, padding='same', activation='relu')
        self.max1 = MaxPooling1D(pool_size=1)
        
        # Second layer: Another Convolution1D followed by max pooling
        self.conv2 = Convolution1D(filters=32, kernel_size=3, padding='same', activation='relu')
        self.max2 = MaxPooling1D(pool_size=1)
        
        # Flatten layer to convert 3D feature maps to 1D feature vectors
        self.flatten = Flatten()
        
        # Dense layer to map to output classes
        self.dense = Dense(num_classes)

    def call(self, inputs):
        # Define forward pass
        x = self.conv1(inputs)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x  # Return predictions

# Define an LSTM-based model that also uses convolutional layers
class LSTM(tf.keras.Model):
    def __init__(self, num_classes):
        super(LSTM, self).__init__()
        
        # Convolutional layer followed by max pooling
        self.conv1 = Convolution1D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.max1 = MaxPooling1D(pool_size=1)
        
        # GRU layer for sequence processing
        self.lstm = GRU(units=30, return_sequences=False)
        
        # Flatten layer to convert 3D feature maps to 1D feature vectors
        self.flatten = Flatten()
        
        # Dense layer to map to output classes
        self.dense = Dense(num_classes)

    def call(self, inputs):
        # Define forward pass
        x = self.conv1(inputs)
        x = self.max1(x)
        x = self.lstm(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x  # Return predictions
