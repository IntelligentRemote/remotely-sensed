#%% pdp
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from keras import layers
import keras
from models import ProbAttention
from keras.models import load_model
import optuna
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from keras.models import Model
import random

# Function to create the first dataset
def create_dataset1(point):
    """
    Creates the first dataset by reading and preprocessing data from an Excel file.
    
    Parameters:
        point (float): Proportion of the data to include in the test split.

    Returns:
        tuple: Returns training and testing datasets (train_X, test_X, train_y, test_y) 
               and full datasets (x, y) after normalization.
    """
    data = pd.read_excel('./data/data.xlsx', header=None)  # Read data from Excel file
    dataf = data.values[1:, 1:]  # Remove headers and the first column
    x = dataf[:, 0:18]  # Features
    y = dataf[:, 18:619]  # Targets
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=point, random_state=42)
    scaler = MinMaxScaler()  # Initialize MinMaxScaler
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)  # Scale training inputs
    test_X = scaler.transform(test_X)    # Scale testing inputs
    x = scaler.transform(x)              # Scale entire input data
    return train_X, test_X, train_y, test_y, x, y

# Function to create the second dataset
def create_dataset2(point):
    """
    Creates the second dataset by reading and preprocessing data from an Excel file.
    
    Parameters:
        point (float): Proportion of the data to include in the test split.

    Returns:
        tuple: Returns training and testing datasets (train_X, test_X, train_y, test_y)
               and full datasets (x, y) after normalization.
    """
    data = pd.read_excel('./data/data.xlsx', header=None)  # Read data from Excel file
    dataf = data.values[1:, 1:]  # Remove headers and the first column
    x = dataf[:, 18:619]  # Features
    y = dataf[:, 619:711] # Targets
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=point, random_state=42)
    scaler = MinMaxScaler()  # Initialize MinMaxScaler
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)  # Scale training inputs
    test_X = scaler.transform(test_X)    # Scale testing inputs
    x = scaler.transform(x)              # Scale entire input data
    return train_X, test_X, train_y, test_y, x, y

# Informer encoder layer
def informer_encoder(inputs, head_size, num_heads, ff_dim, conv_kernel_size, pool_kernel_size, dropout=0):
    """
    Constructs an Informer encoder layer using ProbAttention.
    
    Parameters:
        inputs: Input data for the encoder.
        head_size (int): Size of each attention head.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward layer.
        conv_kernel_size (int): Kernel size for the convolutional layer.
        pool_kernel_size (int): Kernel size for the pooling layer.
        dropout (float): Dropout rate.

    Returns:
        Output tensor after applying the encoder layer.
    """
    x = ProbAttention(head_size, num_heads, dropout, conv_kernel_size, pool_kernel_size, sample_k=20)(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

# Informer decoder layer
def informer_decoder(inputs, head_size, num_heads, ff_dim, conv_kernel_size, pool_kernel_size, dropout=0):
    """
    Constructs an Informer decoder layer using ProbAttention and MultiHeadAttention.
    
    Parameters:
        inputs: Input data for the decoder.
        head_size (int): Size of each attention head.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward layer.
        conv_kernel_size (int): Kernel size for the convolutional layer.
        pool_kernel_size (int): Kernel size for the pooling layer.
        dropout (float): Dropout rate.

    Returns:
        Output tensor after applying the decoder layer.
    """
    x = ProbAttention(head_size, num_heads, dropout, conv_kernel_size, pool_kernel_size, sample_k=20)(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + res

    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

# Model construction function for the first model
def build_model1(input_shape, num_heads, ff_dim, conv_kernel_size, dropout):
    """
    Builds the first Informer model.

    Parameters:
        input_shape (tuple): Shape of the input data.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward network.
        conv_kernel_size (int): Size of the convolutional kernel.
        dropout (float): Dropout rate.

    Returns:
        Keras Model: The constructed Informer model.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(2):  # Two encoder layers
        x = informer_encoder(x, head_size=128, num_heads=num_heads, ff_dim=ff_dim, conv_kernel_size=conv_kernel_size, pool_kernel_size=1, dropout=dropout)
    for _ in range(2):  # Two decoder layers
        x = informer_decoder(x, head_size=128, num_heads=num_heads, ff_dim=ff_dim, conv_kernel_size=conv_kernel_size, pool_kernel_size=1, dropout=dropout)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(601)(x)
    model = keras.Model(inputs, outputs)
    return model

# Model construction function for the second model
def build_model2(input_shape, num_heads, ff_dim, conv_kernel_size, dropout):
    """
    Builds the second Informer model.

    Parameters:
        input_shape (tuple): Shape of the input data.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward network.
        conv_kernel_size (int): Size of the convolutional kernel.
        dropout (float): Dropout rate.

    Returns:
        Keras Model: The constructed Informer model.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(2):  # Two encoder layers
        x = informer_encoder(x, head_size=128, num_heads=num_heads, ff_dim=ff_dim, conv_kernel_size=conv_kernel_size, pool_kernel_size=1, dropout=dropout)
    for _ in range(2):  # Two decoder layers
        x = informer_decoder(x, head_size=128, num_heads=num_heads, ff_dim=ff_dim, conv_kernel_size=conv_kernel_size, pool_kernel_size=1, dropout=dropout)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(92)(x)
    model = keras.Model(inputs, outputs)
    return model

# Objective function for the first model
def objective1(trial):
    """
    Defines the objective function for optimizing the first model using Optuna.

    Parameters:
        trial (optuna.trial.Trial): A trial object from Optuna.

    Returns:
        float: Validation loss.
    """
    # Suggests hyperparameters
    num_heads = trial.suggest_int('num_heads', 2, 8)
    learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.02, 0.05, 0.1, 0.2])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 200, 800)
    
    # Create datasets
    train_X, test_X, train_y, test_y, x, y = create_dataset1(0.3)
    x_train = train_X.reshape((train_X.shape[0], train_X.shape[1], 1)).astype('float64')
    y_train = train_y.astype('float64')
    
    # Initialize KFold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=random.randint(0, 100))
    val_losses = []

    # Train and validate using cross-validation
    for train_index, val_index in kf.split(x_train):
        model1 = build_model1(
            input_shape=x_train.shape[1:], 
            num_heads=num_heads, 
            ff_dim=trial.suggest_int('ff_dim', 4, 64),
            conv_kernel_size=trial.suggest_int('conv_kernel_size', 1, 3),
            dropout=trial.suggest_float('dropout', 0.0, 0.5)
        )
        model1.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["mean_absolute_error"]
        )
        history = model1.fit(
            x_train[train_index], y_train[train_index],
            epochs=epochs, batch_size=batch_size,
            verbose=0,
            validation_data=(x_train[val_index], y_train[val_index])
        )
        val_losses.append(min(history.history['val_loss']))
    
    return np.mean(val_losses)

# Objective function for the second model
def objective2(trial):
    """
    Defines the objective function for optimizing the second model using Optuna.

    Parameters:
        trial (optuna.trial.Trial): A trial object from Optuna.

    Returns:
        float: Validation loss.
    """
    # Suggests hyperparameters
    num_heads = trial.suggest_int('num_heads', 2, 8)
    learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.02, 0.05, 0.1, 0.2])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 200, 800)
    
    # Create datasets
    train_X, test_X, train_y, test_y, x, y = create_dataset2(0.3)
    x_train = train_X.reshape((train_X.shape[0], 1, train_X.shape[1])).astype('float64')
    y_train = train_y.astype('float64')
    
    # Initialize KFold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=random.randint(0, 100))
    val_losses = []

    # Train and validate using cross-validation
    for train_index, val_index in kf.split(x_train):
        model2 = build_model2(
            input_shape=x_train.shape[1:], 
            num_heads=num_heads, 
            ff_dim=4,  
            conv_kernel_size=1,  
            dropout=0.2,
        )
        model2.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["mean_absolute_error"]
        )
        history = model2.fit(
            x_train[train_index], y_train[train_index],
            epochs=epochs, batch_size=batch_size,
            verbose=0,
            validation_data=(x_train[val_index], y_train[val_index])
        )
        val_losses.append(min(history.history['val_loss']))
    
    return np.mean(val_losses)

# Optimization and visualization function
def optimize_and_visualize(objective_function, n_trials=20, direction='minimize'):
    """
    Optimizes the given objective function using Optuna and visualizes the results.
    
    Parameters:
        objective_function (callable): The objective function to optimize.
        n_trials (int): The number of trials for optimization.
        direction (str): Optimization direction ('minimize' or 'maximize').

    Returns:
        dict: The best hyperparameters found.
    """
    with tqdm(total=n_trials, desc="Optimization Progress") as pbar:
        def callback(study, trial):
            pbar.update(1)

        # Creates and optimizes the study
        study = optuna.create_study(direction=direction)
        study.optimize(objective_function, n_trials=n_trials, callbacks=[callback])

    # Visualize optimization results
    optuna.visualization.plot_optimization_history(study)
    plt.show()

    optuna.visualization.plot_param_importances(study)
    plt.show()

    # Return the best parameters
    return study.best_params

# Target function for multiprocessing
def target_function(objective_function, n_trials, direction, queue):
    """
    A global function to execute optimization in a process and place results into a queue.

    Parameters:
        objective_function (callable): The objective function for optimization.
        n_trials (int): The number of optimization trials.
        direction (str): The direction of optimization ('minimize' or 'maximize').
        queue (multiprocessing.Queue): Queue for storing results.
    """
    best_params = optimize_and_visualize(objective_function, n_trials, direction)
    queue.put(best_params)

# Function to run optimization in a separate process
def run_optimization_in_process(objective_function, n_trials, direction):
    # Create and run a process for optimization task and store results in a queue
    queue = multiprocessing.Queue()

    process = multiprocessing.Process(
        target=target_function, 
        args=(objective_function, n_trials, direction, queue)
    )
    process.start()
    process.join()  # Ensure process completion
    best_params = queue.get()  # Retrieve best parameters from queue
    return best_params

# Retrain model with optimal parameters
def retrain_model_with_params(create_dataset_func, build_model_func, best_params):
    """
    Retrains the model using the optimal parameters found in the previous steps.
    
    Parameters:
        create_dataset_func (function): Function to create the dataset.
        build_model_func (function): Function to build the model.
        best_params (dict): Best parameters obtained from optimization.

    Returns:
        Trained Keras Model.
    """
    train_X, test_X, train_y, test_y, x, y = create_dataset_func(0.3)
    x_train = train_X.reshape((train_X.shape[0], -1, train_X.shape[1])).astype('float64')
    y_train = train_y.astype('float64')

    model = build_model_func(input_shape=x_train.shape[1:], **best_params)
    model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(learning_rate=best_params.get('learning_rate', 0.001)),
        metrics=["mean_absolute_error"]
    )
    model.fit(x_train, y_train, epochs=best_params.get('epochs', 100), batch_size=best_params.get('batch_size', 64), verbose=1)

    # Evaluate the model
    test_loss, test_mae = model.evaluate(test_X.reshape((test_X.shape[0], -1, test_X.shape[1])).astype('float64'), test_y.astype('float64'))
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    return model

# Cascade the two models
def cascade_models(model1, model2):
    """
    Cascades two models so that the output of the first model is the input to the second model.
    
    Parameters:
        model1 (Model): The first Keras Model.
        model2 (Model): The second Keras Model.

    Returns:
        Keras Model: The cascaded model.
    """
    from keras.layers import Input, Reshape

    # Ensure output from model1 can be inputted to model2
    input1 = Input(shape=model1.input_shape[1:])
    
    # Get output from the first model
    output1 = model1(input1)
    # Reshape output to fit model2 input
    reshaped_output = Reshape((1, output1.shape[-1]))(output1)
    
    # Input of the second model is the output of the first
    output2 = model2(reshaped_output)
    
    # Construct the combined model
    combined_model = Model(inputs=input1, outputs=output2)
    
    # Compile the combined model if retraining is required
    combined_model.compile(optimizer='adam', loss='mse')
    
    return combined_model

if __name__ == '__main__':
    # Assume objective1 and objective2 are the target functions
    best_params1 = run_optimization_in_process(objective1, n_trials=20, direction='minimize')
    best_params2 = run_optimization_in_process(objective2, n_trials=20, direction='minimize')

    print("Best Parameters for Objective 1:", best_params1)
    print("Best Parameters for Objective 2:", best_params2)

    # Retrain the models with the best parameters obtained
    model1 = retrain_model_with_params(create_dataset1, build_model1, best_params1)
    model2 = retrain_model_with_params(create_dataset2, build_model2, best_params2)
    
    # Cascade the models into one
    combined_model = cascade_models(model1, model2)
    combined_model.save('combined_model.h5')

    # Load the combined model and make predictions
    _, test_X, _, test_y, _, _ = create_dataset1(0.3)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], 1)).astype('float64')
    combined_model = load_model('combined_model.h5',custom_objects={'ProbAttention':ProbAttention})

    # Predict using the combined model
    predict = combined_model.predict(test_X)

    # Save predictions to an Excel file
    with pd.ExcelWriter('predictions.xlsx') as writer:
        for i in range(predict.shape[0]):
            predict_subtable = predict[i].reshape(-1, 1)
            test_y_subtable = test_y[i].reshape(-1, 1)
        
            # Create a DataFrame for each test case
            df = pd.DataFrame(np.hstack((predict_subtable, test_y_subtable)), columns=['Predict', 'Actual'])
        
            # Write each test result to a separate sheet
            sheet_name = f'Subtable_{i+1}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Prediction results have been saved to 'predictions.xlsx'")


#%% pdp
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from keras import layers
import keras
import optuna
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from keras.models import Model
import random

# Function to create the first dataset
def create_dataset1(point):
    """
    Creates the first dataset by reading and processing data from an Excel file.

    Parameters:
        point (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Training and testing input and output data, and full input and output data.
    """
    data = pd.read_excel('./data/data.xlsx', header=None)  # Load data from Excel
    dataf = data.values[1:, 1:]  # Remove headers and the first column
    data = np.array(dataf)
    x = data[:, 0:18]  # Input features
    y = data[:, 18:619]  # Output targets
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=point, random_state=42)
    scaler = MinMaxScaler()  # Initialize scaler
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)  # Scale training input
    test_X = scaler.transform(test_X)    # Scale testing input
    x = scaler.transform(x)              # Scale full input
    return train_X, test_X, train_y, test_y, x, y

# Function to create the second dataset
def create_dataset2(point):
    """
    Creates the second dataset by reading and processing data from an Excel file.

    Parameters:
        point (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Training and testing input and output data, and full input and output data.
    """
    data = pd.read_excel('./data/data.xlsx', header=None)  # Load data from Excel
    dataf = data.values[1:, 1:]  # Remove headers and the first column
    data = np.array(dataf)
    x = data[:, 18:619]  # Input features
    y = data[:, 619:711]  # Output targets
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=point, random_state=42)
    scaler = MinMaxScaler()  # Initialize scaler
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)  # Scale training input
    test_X = scaler.transform(test_X)    # Scale testing input
    x = scaler.transform(x)              # Scale full input
    return train_X, test_X, train_y, test_y, x, y

# Transformer encoder block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    Constructs a transformer-like encoder layer with multi-head attention and feedforward network.

    Parameters:
        inputs: Input tensor to the encoder.
        head_size (int): Dimensionality of each attention head.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feedforward layer.
        dropout (float): Dropout rate.

    Returns:
        Output tensor after applying transformation.
    """
    # Attention and normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed forward network
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

# Model building functions
def build_model1(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    """
    Constructs the first transformer-based model.

    Parameters:
        input_shape (tuple): Shape of the input data.
        head_size (int): Dimensionality of each attention head.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feedforward network.
        num_transformer_blocks (int): Number of transformer blocks.
        mlp_units (list): List of integers representing the units in MLP layers.
        dropout (float): Dropout rate for Transformer layers.
        mlp_dropout (float): Dropout rate for MLP layers.

    Returns:
        Keras Model: Compiled model ready to be trained.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(601)(x)  # Output layer with 601 units
    model = keras.Model(inputs, outputs)
    return model

def build_model2(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    """
    Constructs the second transformer-based model.

    Parameters:
        input_shape (tuple): Shape of the input data.
        head_size (int): Dimensionality of each attention head.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feedforward network.
        num_transformer_blocks (int): Number of transformer blocks.
        mlp_units (list): List of integers representing the units in MLP layers.
        dropout (float): Dropout rate for Transformer layers.
        mlp_dropout (float): Dropout rate for MLP layers.

    Returns:
        Keras Model: Compiled model ready to be trained.
    """
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(92)(x)  # Output layer with 92 units
    model = keras.Model(inputs, outputs)
    return model

# Objective function for Optuna optimization of the first model
def objective1(trial):
    """
    Objective function for optimizing the first model using Optuna.

    Parameters:
        trial (optuna.trial.Trial): Optuna trial object to suggest hyperparameters.

    Returns:
        float: Mean validation loss across cross-validation folds.
    """
    # Suggest hyperparameters
    num_heads = trial.suggest_int('num_heads', 2, 8)
    learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.02, 0.05, 0.1, 0.2])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 200, 800)
    
    # Create the first dataset
    train_X, test_X, train_y, test_y, x, y = create_dataset1(0.3)
    x_train = train_X.reshape((train_X.shape[0], train_X.shape[1], 1)).astype('float64')
    y_train = train_y.astype('float64')
    
    # Cross-validation setup
    kf = KFold(n_splits=10, shuffle=True, random_state=random.randint(0, 100))
    val_losses = []

    # Train and evaluate using cross-validation
    for train_index, val_index in kf.split(x_train):
        model1 = build_model1(
            input_shape=x_train.shape[1:], 
            head_size=128,
            num_heads=num_heads, 
            ff_dim=4,  
            num_transformer_blocks=2,
            mlp_units=[64],
            mlp_dropout=0.2,
            dropout=0.2,
        )
        model1.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["mean_absolute_error"]
        )
        history = model1.fit(
            x_train[train_index], y_train[train_index],
            epochs=epochs, batch_size=batch_size,
            verbose=0,
            validation_data=(x_train[val_index], y_train[val_index])
        )
        val_losses.append(min(history.history['val_loss']))
    
    return np.mean(val_losses)

# Objective function for Optuna optimization of the second model
def objective2(trial):
    """
    Objective function for optimizing the second model using Optuna.

    Parameters:
        trial (optuna.trial.Trial): Optuna trial object to suggest hyperparameters.

    Returns:
        float: Mean validation loss across cross-validation folds.
    """
    # Suggest hyperparameters
    num_heads = trial.suggest_int('num_heads', 2, 8)
    learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.02, 0.05, 0.1, 0.2])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 200, 800)
    
    # Create the second dataset
    train_X, test_X, train_y, test_y, x, y = create_dataset2(0.3)
    x_train = train_X.reshape((train_X.shape[0], 1, train_X.shape[1])).astype('float64')
    y_train = train_y.astype('float64')
    
    # Cross-validation setup
    kf = KFold(n_splits=10, shuffle=True, random_state=random.randint(0, 100))
    val_losses = []

    # Train and evaluate using cross-validation
    for train_index, val_index in kf.split(x_train):
        model2 = build_model2(
            input_shape=x_train.shape[1:], 
            head_size=128,
            num_heads=num_heads, 
            ff_dim=4,  
            num_transformer_blocks=2,
            mlp_units=[64],
            mlp_dropout=0.2,
            dropout=0.2,
        )
        model2.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["mean_absolute_error"]
        )
        history = model2.fit(
            x_train[train_index], y_train[train_index],
            epochs=epochs, batch_size=batch_size,
            verbose=0,
            validation_data=(x_train[val_index], y_train[val_index])
        )
        val_losses.append(min(history.history['val_loss']))
    
    return np.mean(val_losses)

# Optimize hyperparameters and visualize results
def optimize_and_visualize(objective_function, n_trials=20, direction='minimize'):
    """
    Uses Optuna to perform hyperparameter optimization and visualize results.

    Parameters:
        objective_function (callable): The objective function to optimize.
        n_trials (int): Number of optimization trials. Default is 20.
        direction (str): Direction of optimization ('minimize' or 'maximize').

    Returns:
        dict: Best hyperparameters dictionary.
    """
    with tqdm(total=n_trials, desc="Optimization Progress") as pbar:
        def callback(study, trial):
            pbar.update(1)

        # Create and optimize an Optuna study
        study = optuna.create_study(direction=direction)
        study.optimize(objective_function, n_trials=n_trials, callbacks=[callback])

    # Visualize optimization results
    optuna.visualization.plot_optimization_history(study)
    plt.show()

    optuna.visualization.plot_param_importances(study)
    plt.show()

    # Return the best parameters
    return study.best_params

# Retrain models with the best parameters
def retrain_model_with_params(create_dataset_func, build_model_func, best_params):
    """
    Retrains a model using the best parameters obtained from optimization.

    Parameters:
        create_dataset_func (function): Function to create the dataset.
        build_model_func (function): Function to build the model.
        best_params (dict): Best parameters obtained from optimization.

    Returns:
        Keras Model: Trained model.
    """
    train_X, test_X, train_y, test_y, x, y = create_dataset_func(0.3)
    x_train = train_X.reshape((train_X.shape[0], -1, train_X.shape[1])).astype('float64')
    y_train = train_y.astype('float64')

    model = build_model_func(input_shape=x_train.shape[1:], **best_params)
    model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(learning_rate=best_params.get('learning_rate', 0.001)),
        metrics=["mean_absolute_error"]
    )
    model.fit(x_train, y_train, epochs=best_params.get('epochs', 100), batch_size=best_params.get('batch_size', 64), verbose=1)

    # Evaluate the model
    test_loss, test_mae = model.evaluate(test_X.reshape((test_X.shape[0], -1, test_X.shape[1])).astype('float64'), test_y.astype('float64'))
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    return model

# Cascade two models
def cascade_models(model1, model2):
    """
    Cascades two models so that the output of the first model is used as input to the second model.

    Parameters:
        model1 (Keras Model): The first model.
        model2 (Keras Model): The second model.

    Returns:
        Keras Model: Combined model.
    """
    from keras.layers import Input, Reshape

    # Ensure that the output of model1 matches the input shape of model2
    input1 = Input(shape=model1.input_shape[1:])
    
    # Get the first model's output
    output1 = model1(input1)
    # Reshape the output to fit the second model's input
    reshaped_output = Reshape((1, output1.shape[-1]))(output1)
    
    # Use the first model's output as the second model's input
    output2 = model2(reshaped_output)
    
    # Build the complete cascaded model
    combined_model = Model(inputs=input1, outputs=output2)
    
    # Compile if you need to retrain the model
    combined_model.compile(optimizer='adam', loss='mse')
    
    return combined_model

# Main function to optimize and train models
if __name__ == '__main__':
    # Optimize the first model
    best_params1 = optimize_and_visualize(objective1, n_trials=20, direction='minimize')
    print("Best Parameters for Objective 1:", best_params1)
    
    # Optimize the second model
    best_params2 = optimize_and_visualize(objective2, n_trials=20, direction='minimize')
    print("Best Parameters for Objective 2:", best_params2)

    # Retrain the models with the best parameters obtained
    model1 = retrain_model_with_params(create_dataset1, build_model1, best_params1)
    model2 = retrain_model_with_params(create_dataset2, build_model2, best_params2)
    
    # Combine the models into one
    combined_model = cascade_models(model1, model2)
    combined_model.save('combined_model.h5')
    
    # Predict with the combined model
    _, test_X, _, test_y, _, _ = create_dataset1(0.3)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], 1)).astype('float64')
    combined_model = load_model('combined_model.h5')

    # Perform predictions
    predict = combined_model.predict(test_X)

    # Save predictions to an Excel file
    with pd.ExcelWriter('predictions.xlsx') as writer:
        for i in range(predict.shape[0]):
            predict_subtable = predict[i].reshape(-1, 1)
            test_y_subtable = test_y[i].reshape(-1, 1)
        
            # Create a DataFrame for each sub-table of predictions and actual values
            df = pd.DataFrame(np.hstack((predict_subtable, test_y_subtable)), columns=['Predict', 'Actual'])
        
            # Write the DataFrame to a separate Excel sheet
            sheet_name = f'Subtable_{i+1}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Prediction results have been saved to 'predictions.xlsx'")



# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from keras import layers
import tensorflow as tf
import keras
from models import CNN  # Import a custom CNN model definition - ensure this file/class exists
import optuna
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar
import multiprocessing
from keras.models import Model
import random

# Function to create the first dataset
def create_dataset1(point):
    """
    Create the first dataset by reading, processing, and splitting data from an Excel file.

    Parameters:
        point (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Training and testing data (both features and targets) and the full-scale dataset.
    """
    data = pd.read_excel('./data/data.xlsx', header=None)  # Load data from an Excel file
    dataf = data.values[1:, 1:]  # Skip the header and the first column
    data = np.array(dataf)
    x = data[:, 0:18]  # Select input features
    y = data[:, 18:619]  # Select output targets
    # Split into training and testing datasets
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=point, random_state=42)
    scaler = MinMaxScaler()  # Initialize Min-Max Scaler
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)  # Normalize training inputs
    test_X = scaler.transform(test_X)    # Normalize testing inputs
    x = scaler.transform(x)              # Normalize all inputs
    return train_X, test_X, train_y, test_y, x, y

# Function to create the second dataset
def create_dataset2(point):
    """
    Create the second dataset by reading, processing, and splitting data from an Excel file.

    Parameters:
        point (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Training and testing data (both features and targets) and the full-scale dataset.
    """
    data = pd.read_excel('./data/data.xlsx', header=None)  # Load data from an Excel file
    dataf = data.values[1:, 1:]  # Skip the header and the first column
    data = np.array(dataf)
    x = data[:, 18:619]  # Select input features
    y = data[:, 619:711]  # Select output targets
    # Split into training and testing datasets
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=point, random_state=42)
    scaler = MinMaxScaler()  # Initialize Min-Max Scaler
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)  # Normalize training inputs
    test_X = scaler.transform(test_X)    # Normalize testing inputs
    x = scaler.transform(x)              # Normalize all inputs
    return train_X, test_X, train_y, test_y, x, y

# Objective function for Optuna optimization - Model 1
def objective1(trial):
    """
    Objective function for optimizing the first model using Optuna.

    Parameters:
        trial (optuna.trial.Trial): An Optuna trial object to suggest hyperparameters.

    Returns:
        float: Mean validation loss across cross-validation folds.
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.02, 0.05, 0.1, 0.2])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 200, 800)
    
    # Create the first dataset
    train_X, test_X, train_y, test_y, x, y = create_dataset1(0.3)
    x_train = train_X.reshape((train_X.shape[0], train_X.shape[1], 1)).astype('float64')
    y_train = train_y.astype('float64')
    
    # K-Fold Cross Validation
    kf = KFold(n_splits=10, shuffle=True, random_state=random.randint(0, 100))
    val_losses = []

    # Train and validate using cross-validation
    for train_index, val_index in kf.split(x_train):
        # Initialize the model using the CNN module (assuming CNN is a custom class returning model layers)
        model1 = tf.keras.Sequential(CNN(601))
        model1.compile(
            loss='mse',  # Mean Squared Error loss
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["mean_absolute_error"]  # Additional metric for evaluation
        )
        history = model1.fit(
            x_train[train_index], y_train[train_index],
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_data=(x_train[val_index], y_train[val_index])  # Validation data from the split
        )
        val_losses.append(min(history.history['val_loss']))  # Record minimum validation loss
    
    return np.mean(val_losses)  # Return average validation loss

# Objective function for Optuna optimization - Model 2
def objective2(trial):
    """
    Objective function for optimizing the second model using Optuna.

    Parameters:
        trial (optuna.trial.Trial): An Optuna trial object to suggest hyperparameters.

    Returns:
        float: Mean validation loss across cross-validation folds.
    """
    learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.02, 0.05, 0.1, 0.2])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 200, 800)
    
    # Create the second dataset
    train_X, test_X, train_y, test_y, x, y = create_dataset2(0.3)
    x_train = train_X.reshape((train_X.shape[0], 1, train_X.shape[1])).astype('float64')
    y_train = train_y.astype('float64')
    
    # K-Fold Cross Validation
    kf = KFold(n_splits=10, shuffle=True, random_state=random.randint(0, 100))
    val_losses = []

    # Train and validate using cross-validation
    for train_index, val_index in kf.split(x_train):
        # Initialize the model using the CNN module
        model2 = tf.keras.Sequential(CNN(92))
        model2.compile(
            loss='mse',  # Mean Squared Error loss
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["mean_absolute_error"]  # Additional metric for evaluation
        )
        history = model2.fit(
            x_train[train_index], y_train[train_index],
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_data=(x_train[val_index], y_train[val_index])  # Validation data from the split
        )
        val_losses.append(min(history.history['val_loss']))  # Record minimum validation loss
    
    return np.mean(val_losses)  # Return average validation loss

# Optimize and visualize results
def optimize_and_visualize(objective_function, n_trials=20, direction='minimize'):
    """
    Use Optuna to perform hyperparameter optimization and visualize results.

    Parameters:
        objective_function (callable): The objective function to optimize.
        n_trials (int): Number of optimization trials. Default is 20.
        direction (str): Optimization direction ('minimize' or 'maximize').

    Returns:
        dict: Best hyperparameters dictionary.
    """
    with tqdm(total=n_trials, desc="Optimization Progress") as pbar:
        def callback(study, trial):
            pbar.update(1)  # Update progress bar

        # Create and optimize an Optuna study
        study = optuna.create_study(direction=direction)
        study.optimize(objective_function, n_trials=n_trials, callbacks=[callback])

    # Visualize optimization results
    optuna.visualization.plot_optimization_history(study)
    plt.show()

    optuna.visualization.plot_param_importances(study)
    plt.show()

    # Return the best parameters
    return study.best_params

# Retrain model with optimal parameters
def retrain_model_with_params(create_dataset_func, build_model_func, best_params):
    """
    Retrain a model using the best parameters obtained from optimization.

    Parameters:
        create_dataset_func (function): Function to create the dataset.
        build_model_func (function): Function to build the model.
        best_params (dict): Best parameters obtained from optimization.

    Returns:
        Keras Model: Trained model.
    """
    train_X, test_X, train_y, test_y, x, y = create_dataset_func(0.3)
    x_train = train_X.reshape((train_X.shape[0], -1, train_X.shape[1])).astype('float64')
    y_train = train_y.astype('float64')

    model = build_model_func(input_shape=x_train.shape[1:], **best_params)  # Model constructed with best hyperparameters
    model.compile(
        loss='mse',  # Use Mean Squared Error loss
        optimizer=keras.optimizers.Adam(learning_rate=best_params.get('learning_rate', 0.001)),
        metrics=["mean_absolute_error"]  # Use Mean Absolute Error for metric
    )
    model.fit(x_train, y_train, epochs=best_params.get('epochs', 100), batch_size=best_params.get('batch_size', 64), verbose=1)

    # Evaluate the model
    test_loss, test_mae = model.evaluate(test_X.reshape((test_X.shape[0], -1, test_X.shape[1])).astype('float64'), test_y.astype('float64'))
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    return model  # Return the trained model

# Combine or "cascade" models
def cascade_models(model1, model2):
    """
    Cascade two models so that the output of the first model is used as input to the second model.

    Parameters:
        model1 (Keras Model): The first Keras model.
        model2 (Keras Model): The second Keras model.

    Returns:
        Keras Model: Combined model.
    """
    from keras.layers import Input, Reshape

    # Ensure that model1's output shape matches model2's input shape
    input1 = Input(shape=model1.input_shape[1:])
    
    # Get the output from the first model
    output1 = model1(input1)
    # Reshape to ensure proper input shape for the second model
    reshaped_output = Reshape((1, output1.shape[-1]))(output1)
    
    # Pass reshaped output to the second model
    output2 = model2(reshaped_output)
    
    # Construct the full cascaded model
    combined_model = Model(inputs=input1, outputs=output2)
    
    # Compile the combined model if retraining is necessary
    combined_model.compile(optimizer='adam', loss='mse')
    
    return combined_model

# Main execution
if __name__ == '__main__':
    # Optimize the first and second models
    best_params1 = optimize_and_visualize(objective1, n_trials=20, direction='minimize')
    print("Best Parameters for Objective 1:", best_params1)
    
    best_params2 = optimize_and_visualize(objective2, n_trials=20, direction='minimize')
    print("Best Parameters for Objective 2:", best_params2)

    # Retrain the models with the best parameters obtained from optimization
    model1 = retrain_model_with_params(create_dataset1, lambda input_shape, **params: tf.keras.Sequential(CNN(601)), best_params1)
    model2 = retrain_model_with_params(create_dataset2, lambda input_shape, **params: tf.keras.Sequential(CNN(92)), best_params2)
    
    # Combine models into one
    combined_model = cascade_models(model1, model2)
    combined_model.save('combined_model.h5')

    # Load the model and prepare test data
    _, test_X, _, test_y, _, _ = create_dataset1(0.3)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], 1)).astype('float64')
    combined_model = load_model('combined_model.h5')

    # Use the model to predict
    predict = combined_model.predict(test_X)

    # Create an Excel writer using Pandas
    with pd.ExcelWriter('predictions.xlsx') as writer:
        for i in range(predict.shape[0]):
            # Extract data for current prediction
            predict_subtable = predict[i].reshape(-1, 1)
            test_y_subtable = test_y[i].reshape(-1, 1)
        
            # Create a DataFrame for current data
            df = pd.DataFrame(np.hstack((predict_subtable, test_y_subtable)), columns=['Predict', 'Actual'])
        
            # Write the DataFrame to a specific Excel sheet
            sheet_name = f'Subtable_{i+1}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Prediction results have been saved to 'predictions.xlsx'")


# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
from keras import layers
import tensorflow as tf
import keras
from models import LSTM  # Assumes LSTM is a custom model class or function defined in models
import optuna
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bars
import multiprocessing
from keras.models import Model
import random

# Function to create the first dataset
def create_dataset1(point):
    """
    Create the first dataset by reading, processing, and splitting data from an Excel file.

    Parameters:
        point (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Training and testing data (features and targets), and full-scale dataset.
    """
    data = pd.read_excel('./data/data.xlsx', header=None)  # Load data from an Excel file without headers
    dataf = data.values[1:, 1:]  # Skip header and first column (could be id or labels)
    data = np.array(dataf)
    
    # Define features and targets for Model 1
    x = data[:, 0:18]  # Features from columns 0 to 17
    y = data[:, 18:619]  # Targets from columns 18 to 618
    
    # Split into training and testing data
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=point, random_state=42)
    
    # Use MinMaxScaler to scale features to the range [0, 1]
    scaler = MinMaxScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    x = scaler.transform(x)
    
    return train_X, test_X, train_y, test_y, x, y

# Function to create the second dataset
def create_dataset2(point):
    """
    Create the second dataset by reading, processing, and splitting data from an Excel file.

    Parameters:
        point (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Training and testing data (features and targets), and full-scale dataset.
    """
    data = pd.read_excel('./data/data.xlsx', header=None)  # Load data from an Excel file without headers
    dataf = data.values[1:, 1:]  # Skip header and first column
    data = np.array(dataf)
    
    # Define features and targets for Model 2
    x = data[:, 18:619]  # Features from columns 18 to 618
    y = data[:, 619:711]  # Targets from columns 619 to 710
    
    # Split into training and testing data
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=point, random_state=42)
    
    # Use MinMaxScaler to scale features to the range [0, 1]
    scaler = MinMaxScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    x = scaler.transform(x)
    
    return train_X, test_X, train_y, test_y, x, y

# Objective function for Optuna optimization - Model 1
def objective1(trial):
    """
    Define the objective function for Model 1 optimization using Optuna.

    Parameters:
        trial (optuna.trial.Trial): An Optuna trial object to suggest hyperparameters.

    Returns:
        float: Mean validation loss across cross-validation folds.
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.02, 0.05, 0.1, 0.2])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 200, 800)
    
    # Create the first dataset
    train_X, test_X, train_y, test_y, x, y = create_dataset1(0.3)
    x_train = train_X.reshape((train_X.shape[0], train_X.shape[1], 1)).astype('float64')  # Reshaping for LSTM
    y_train = train_y.astype('float64')
    
    # Setup K-Fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=random.randint(0, 100))
    val_losses = []

    # Loop over each fold in cross-validation
    for train_index, val_index in kf.split(x_train):
        # Initialize the model with LSTM, assuming LSTM returns layers suitable for Sequential model
        model1 = tf.keras.Sequential(LSTM(601))
        model1.compile(
            loss='mse',  # Mean Squared Error loss
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["mean_absolute_error"]  # Additional metric for evaluation
        )
        
        # Train the model
        history = model1.fit(
            x_train[train_index], y_train[train_index],
            epochs=epochs, batch_size=batch_size,
            verbose=0,
            validation_data=(x_train[val_index], y_train[val_index])  # Validation data from the split
        )
        val_losses.append(min(history.history['val_loss']))  # Record minimum validation loss
    
    return np.mean(val_losses)  # Return mean validation loss

# Objective function for Optuna optimization - Model 2
def objective2(trial):
    """
    Define the objective function for Model 2 optimization using Optuna.

    Parameters:
        trial (optuna.trial.Trial): An Optuna trial object to suggest hyperparameters.

    Returns:
        float: Mean validation loss across cross-validation folds.
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.02, 0.05, 0.1, 0.2])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 200, 800)
    
    # Create the second dataset
    train_X, test_X, train_y, test_y, x, y = create_dataset2(0.3)
    x_train = train_X.reshape((train_X.shape[0], 1, train_X.shape[1])).astype('float64')  # Reshaping for LSTM
    y_train = train_y.astype('float64')
    
    # Setup K-Fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=random.randint(0, 100))
    val_losses = []

    # Loop over each fold in cross-validation
    for train_index, val_index in kf.split(x_train):
        # Initialize the model with LSTM, assuming LSTM returns layers suitable for Sequential model
        model2 = tf.keras.Sequential(LSTM(92))
        model2.compile(
            loss='mse',  # Mean Squared Error loss
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["mean_absolute_error"]  # Additional metric for evaluation
        )
        
        # Train the model
        history = model2.fit(
            x_train[train_index], y_train[train_index],
            epochs=epochs, batch_size=batch_size,
            verbose=0,
            validation_data=(x_train[val_index], y_train[val_index])  # Validation data from the split
        )
        val_losses.append(min(history.history['val_loss']))  # Record minimum validation loss
    
    return np.mean(val_losses)  # Return mean validation loss

# Use Optuna to optimize and visualize hyperparameter tuning
def optimize_and_visualize(objective_function, n_trials=20, direction='minimize'):
    """
    Perform hyperparameter optimization using Optuna and visualize the results.

    Parameters:
        objective_function (callable): The objective function for the optimization.
        n_trials (int): Number of optimization trials. Default is 20.
        direction (str): Direction of optimization ('minimize' or 'maximize').

    Returns:
        dict: Best hyperparameters found during optimization.
    """
    with tqdm(total=n_trials, desc="Optimization Progress") as pbar:
        def callback(study, trial):
            pbar.update(1)  # Update the progress bar for each trial

        # Create and optimize an Optuna study
        study = optuna.create_study(direction=direction)
        study.optimize(objective_function, n_trials=n_trials, callbacks=[callback])

    # Visualize optimization history and parameter importance
    optuna.visualization.plot_optimization_history(study)
    plt.show()

    optuna.visualization.plot_param_importances(study)
    plt.show()

    return study.best_params  # Return the best hyperparameters found

# Retrain the model with optimal parameters found by Optuna
def retrain_model_with_params(create_dataset_func, build_model_func, best_params):
    """
    Retrain a model using the best hyperparameters obtained from Optuna optimization.

    Parameters:
        create_dataset_func (function): Function to create and prepare the dataset.
        build_model_func (function): Function to construct the model architecture.
        best_params (dict): Best hyperparameters for model retraining.

    Returns:
        Keras Model: The trained model ready for evaluation and predictions.
    """
    train_X, test_X, train_y, test_y, x, y = create_dataset_func(0.3)  # Create dataset using specified function
    x_train = train_X.reshape((train_X.shape[0], -1, train_X.shape[1])).astype('float64')  # Reshape for LSTM input
    y_train = train_y.astype('float64')

    # Construct the model using the best parameters found
    model = build_model_func(input_shape=x_train.shape[1:], **best_params)
    model.compile(
        loss='mse',  # Use Mean Squared Error for loss
        optimizer=keras.optimizers.Adam(learning_rate=best_params.get('learning_rate', 0.001)),
        metrics=["mean_absolute_error"]  # Evaluate model performance with Mean Absolute Error
    )
    
    # Train the model with best hyperparameters
    model.fit(x_train, y_train, epochs=best_params.get('epochs', 100), batch_size=best_params.get('batch_size', 64), verbose=1)

    # Evaluate the trained model on the test dataset
    test_loss, test_mae = model.evaluate(test_X.reshape((test_X.shape[0], -1, test_X.shape[1])).astype('float64'), test_y.astype('float64'))
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    return model  # Return the retrained model

# Combine two models where the output of the first model is fed into the second model
def cascade_models(model1, model2):
    """
    Cascade two models by feeding the output of the first model into the second model.

    Parameters:
        model1 (Keras Model): The first Keras model in the cascade.
        model2 (Keras Model): The second Keras model in the cascade.

    Returns:
        Keras Model: Combined model.
    """
    from keras.layers import Input, Reshape

    # Define input shape for the combined model
    input1 = Input(shape=model1.input_shape[1:])
    
    # Generate the output of the first model
    output1 = model1(input1)
    # Reshape the output to fit the input requirements of the second model
    reshaped_output = Reshape((1, output1.shape[-1]))(output1)
    
    # Use the reshaped output as input to the second model
    output2 = model2(reshaped_output)
    
    # Construct the final combined model
    combined_model = Model(inputs=input1, outputs=output2)
    
    # Compile the combined model if retraining or additional evaluation is needed
    combined_model.compile(optimizer='adam', loss='mse')
    
    return combined_model

# Main script execution
if __name__ == '__main__':
    # Hyperparameter optimization for Model 1
    best_params1 = optimize_and_visualize(objective1, n_trials=20, direction='minimize')
    print("Best Parameters for Objective 1:", best_params1)
    
    # Hyperparameter optimization for Model 2
    best_params2 = optimize_and_visualize(objective2, n_trials=20, direction='minimize')
    print("Best Parameters for Objective 2:", best_params2)

    # Retrain models with best parameters obtained from optimization
    model1 = retrain_model_with_params(create_dataset1, lambda input_shape, **params: tf.keras.Sequential(LSTM(601)), best_params1)
    model2 = retrain_model_with_params(create_dataset2, lambda input_shape, **params: tf.keras.Sequential(LSTM(92)), best_params2)
    
    # Cascade the two models into one combined model
    combined_model = cascade_models(model1, model2)
    combined_model.save('combined_model.h5')  # Save the combined model to a file

    # Load the combined model and prepare test data for predictions
    _, test_X, _, test_y, _, _ = create_dataset1(0.3)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], 1)).astype('float64')
    combined_model = load_model('combined_model.h5')

    # Predict using the combined model
    predict = combined_model.predict(test_X)

    # Save predictions and actual values to an Excel file
    with pd.ExcelWriter('predictions.xlsx') as writer:
        for i in range(predict.shape[0]):
            # Reshape prediction and actual data for current test example
            predict_subtable = predict[i].reshape(-1, 1)
            test_y_subtable = test_y[i].reshape(-1, 1)
        
            # Create a DataFrame to store predictions and actual values
            df = pd.DataFrame(np.hstack((predict_subtable, test_y_subtable)), columns=['Predict', 'Actual'])
        
            # Write the DataFrame to an Excel sheet
            sheet_name = f'Subtable_{i+1}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Prediction results have been saved to 'predictions.xlsx'")



# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
import optuna
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import xgboost as xgb
import random
import pickle

# Function to create the first dataset
def create_dataset1(point):
    """
    Create the first dataset by loading, processing, and scaling data from an Excel file.

    Parameters:
        point (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Scaled training and testing data (features and targets), and full-scale dataset.
    """
    data = pd.read_excel('./data/data.xlsx', header=None)  # Load data from Excel file without headers
    dataf = data.values[1:, 1:]  # Skip header and first column
    data = np.array(dataf)
    
    # Split data into features and targets for Model 1
    x = data[:, 0:18]  # Features from columns 0 to 17
    y = data[:, 18:619]  # Targets from columns 18 to 618
    
    # Split into training and testing data
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=point, random_state=42)
    
    # Initialize and fit MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train_X)
    
    # Scale features
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    x = scaler.transform(x)
    
    return train_X, test_X, train_y, test_y, x, y

# Function to create the second dataset
def create_dataset2(point):
    """
    Create the second dataset by loading, processing, and scaling data from an Excel file.

    Parameters:
        point (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Scaled training and testing data (features and targets), and full-scale dataset.
    """
    data = pd.read_excel('./data/data.xlsx', header=None)  # Load data from Excel file without headers
    dataf = data.values[1:, 1:]  # Skip header and first column
    data = np.array(dataf)
    
    # Split data into features and targets for Model 2
    x = data[:, 18:619]  # Features from columns 18 to 618
    y = data[:, 619:711]  # Targets from columns 619 to 710
    
    # Split into training and testing data
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=point, random_state=42)
    
    # Initialize and fit MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train_X)
    
    # Scale features
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    x = scaler.transform(x)
    
    return train_X, test_X, train_y, test_y, x, y

# Objective function for Optuna optimization - Model 1
def objective1(trial):
    """
    Define the optimization objective function for Model 1 using Optuna.

    Parameters:
        trial (optuna.trial.Trial): An Optuna trial object to suggest hyperparameters.

    Returns:
        float: Mean validation loss (mean squared error) across cross-validation folds.
    """
    # Suggest hyperparameters
    n_estimators = trial.suggest_categorical('n_estimators', list(range(50, 301, 50)))
    learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.02, 0.05, 0.1, 0.2])
    max_depth = trial.suggest_categorical('max_depth', [3, 4, 5, 6, 7, 8])
    
    # Create the first dataset
    train_X, test_X, train_y, test_y, x, y = create_dataset1(0.3)
    x_train = train_X.astype('float64')  # Convert to float64 for precision
    y_train = train_y.astype('float64')
    
    # Prepare K-Fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=random.randint(0, 100))
    val_losses = []

    # Loop over each fold
    for train_index, val_index in kf.split(x_train):
        # Split data into training and validation sets
        x_train_split, x_val_split = x_train[train_index], x_train[val_index]
        y_train_split, y_val_split = y_train[train_index], y_train[val_index]
    
        # Initialize and train XGBoost model
        model1 = xgb.XGBRegressor(max_depth=max_depth,
                                  learning_rate=learning_rate,
                                  n_estimators=n_estimators)
    
        model1.fit(x_train_split, y_train_split)
    
        # Predict and calculate validation loss (MSE)
        y_val_pred = model1.predict(x_val_split)
        val_loss = mean_squared_error(y_val_split, y_val_pred)
    
        # Append current fold validation loss to list
        val_losses.append(val_loss)
    
    return np.mean(val_losses)  # Return mean validation loss

# Objective function for Optuna optimization - Model 2
def objective2(trial):
    """
    Define the optimization objective function for Model 2 using Optuna.

    Parameters:
        trial (optuna.trial.Trial): An Optuna trial object to suggest hyperparameters.

    Returns:
        float: Mean validation loss (mean squared error) across cross-validation folds.
    """
    # Suggest hyperparameters
    n_estimators = trial.suggest_categorical('n_estimators', list(range(50, 301, 50)))
    learning_rate = trial.suggest_categorical('learning_rate', [0.01, 0.02, 0.05, 0.1, 0.2])
    max_depth = trial.suggest_categorical('max_depth', [3, 4, 5, 6, 7, 8])
    
    # Create the second dataset
    train_X, test_X, train_y, test_y, x, y = create_dataset2(0.3)
    x_train = train_X.astype('float64')  # Convert to float64 for precision
    y_train = train_y.astype('float64')
    
    # Prepare K-Fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=random.randint(0, 100))
    val_losses = []

    # Loop over each fold
    for train_index, val_index in kf.split(x_train):
        # Split data into training and validation sets
        x_train_split, x_val_split = x_train[train_index], x_train[val_index]
        y_train_split, y_val_split = y_train[train_index], y_train[val_index]
    
        # Initialize and train XGBoost model
        model2 = xgb.XGBRegressor(max_depth=max_depth,
                                  learning_rate=learning_rate,
                                  n_estimators=n_estimators)
    
        model2.fit(x_train_split, y_train_split)
    
        # Predict and calculate validation loss (MSE)
        y_val_pred = model2.predict(x_val_split)
        val_loss = mean_squared_error(y_val_split, y_val_pred)
    
        # Append current fold validation loss to list
        val_losses.append(val_loss)
    
    return np.mean(val_losses)  # Return mean validation loss

# Optimize hyperparameters using Optuna and visualize results
def optimize_and_visualize(objective_function, n_trials=20, direction='minimize'):
    """
    Perform hyperparameter optimization using Optuna and visualize the results.

    Parameters:
        objective_function (callable): The objective function for optimization.
        n_trials (int): Number of optimization trials. Default is 20.
        direction (str): Direction of optimization ('minimize' or 'maximize').

    Returns:
        dict: Best hyperparameters found during optimization.
    """
    with tqdm(total=n_trials, desc="Optimization Progress") as pbar:
        def callback(study, trial):
            pbar.update(1)  # Update progress bar for each trial

        # Create and optimize an Optuna study
        study = optuna.create_study(direction=direction)
        study.optimize(objective_function, n_trials=n_trials, callbacks=[callback])

    # Visualize optimization results
    optuna.visualization.plot_optimization_history(study)
    plt.show()

    optuna.visualization.plot_param_importances(study)
    plt.show()

    return study.best_params  # Return the best parameters

# Retrain the XGBoost model with the best parameters obtained from Optuna
def retrain_xgb_with_params(create_dataset_func, best_params):
    """
    Retrain an XGBoost model using the best hyperparameters obtained from Optuna optimization.

    Parameters:
        create_dataset_func (function): Function to create and prepare the dataset.
        best_params (dict): Best hyperparameters for model retraining.

    Returns:
        XGBRegressor: The trained XGBoost model.
    """
    # Create and prepare the dataset
    train_X, test_X, train_y, test_y, x, y = create_dataset_func(0.1)
    x_train = train_X.astype('float64')  # Ensure data is in float64 format for precision
    y_train = train_y.astype('float64')

    # Initialize model with best hyperparameters
    model = xgb.XGBRegressor(
        max_depth=best_params.get('max_depth', 3),
        learning_rate=best_params.get('learning_rate', 0.1),
        n_estimators=best_params.get('n_estimators', 100),
        verbosity=1
    )

    # Train the model
    model.fit(x_train, y_train, eval_set=[(x_train, y_train)], verbose=True)

    # Evaluate the model
    test_prediction = model.predict(test_X.reshape((test_X.shape[0], -1)).astype('float64'))
    test_loss = mean_squared_error(test_y, test_prediction)
    print(f"Test Loss: {test_loss}")

    return model  # Return the trained model

# Class for handling cascaded models
class CascadedModel:
    def __init__(self, model1, model2):
        """
        Initialize a cascaded model by combining two models.

        Parameters:
            model1 (XGBRegressor): The first model in the cascade.
            model2 (XGBRegressor): The second model in the cascade.
        """
        self.model1 = model1
        self.model2 = model2

    def predict(self, X):
        """
        Perform predictions using the cascaded models.

        Parameters:
            X (np.ndarray): Input features for predictions.

        Returns:
            np.ndarray: The final predictions from the cascaded model.
        """
        # Perform prediction with the first model
        intermediate_prediction = self.model1.predict(X)
        
        # Use first model's output as input features for second model
        intermediate_prediction = intermediate_prediction.reshape(-1, 1)  # Ensure output is 2D
        final_prediction = self.model2.predict(intermediate_prediction)
        
        return final_prediction

    def save_model(self, filename):
        """
        Save the cascaded model to a file.

        Parameters:
            filename (str): The file path to save the model.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        """
        Load a cascaded model from a file.

        Parameters:
            filename (str): The file path from which to load the model.

        Returns:
            CascadedModel: The loaded cascaded model instance.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

# Main script execution
if __name__ == '__main__':
    # Optimize and find the best parameters using Optuna
    best_params1 = optimize_and_visualize(objective1, n_trials=20, direction='minimize')
    print("Best Parameters for Objective 1:", best_params1)
    
    best_params2 = optimize_and_visualize(objective2, n_trials=20, direction='minimize')
    print("Best Parameters for Objective 2:", best_params2)

    # Retrain model with the obtained optimal hyperparameters
    model1 = retrain_xgb_with_params(create_dataset1, best_params1)
    model2 = retrain_xgb_with_params(create_dataset2, best_params2)

    # Create and save a cascaded model
    cascaded_model = CascadedModel(model1, model2)
    cascaded_model.save_model('cascaded_xgb_model.pkl')

    # Load cascaded model from file and perform predictions
    loaded_cascaded_model = CascadedModel.load_model('cascaded_xgb_model.pkl')
    _, test_X, _, test_y, _, _ = create_dataset1(0.3)
    test_X = test_X.reshape((test_X.shape[0], -1)).astype('float64')

    predict = loaded_cascaded_model.predict(test_X)

    # Use Pandas ExcelWriter to save predictions
    with pd.ExcelWriter('predictions.xlsx') as writer:
        for i in range(predict.shape[0]):
            # Reshape predictions and actual values for output
            predict_subtable = predict[i].reshape(-1, 1)
            test_y_subtable = test_y[i].reshape(-1, 1)
        
            # Create DataFrame for the current prediction vs. actual values
            df = pd.DataFrame(np.hstack((predict_subtable, test_y_subtable)), columns=['Predict', 'Actual'])
        
            # Write DataFrame to a specific Excel sheet
            sheet_name = f'Subtable_{i+1}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Prediction results have been saved to 'predictions.xlsx'")



# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
import optuna
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import random
import pickle

# Function to create the first dataset
def create_dataset1(point):
    """
    Create the first dataset by loading, processing, and splitting data from an Excel file.

    Parameters:
        point (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Scaled training and testing data (features and targets), and full-scale dataset.
    """
    data = pd.read_excel('./data/data.xlsx', header=None)  # Load data from Excel file without headers
    dataf = data.values[1:, 1:]  # Skip the first row and column (usually headers or non-data parts)
    data = np.array(dataf)
    
    # Define features and targets for Model 1
    x = data[:, 0:18]  # Features from columns 0 to 17
    y = data[:, 18:619]  # Targets from columns 18 to 618
    
    # Split dataset into training and testing sets
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=point, random_state=42)
    
    # Initialize and fit the MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train_X)
    
    # Scale the datasets
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    x = scaler.transform(x)
    
    return train_X, test_X, train_y, test_y, x, y

# Function to create the second dataset
def create_dataset2(point):
    """
    Create the second dataset by loading, processing, and splitting data from an Excel file.

    Parameters:
        point (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Scaled training and testing data (features and targets), and full-scale dataset.
    """
    data = pd.read_excel('./data/data.xlsx', header=None)  # Load data from Excel file without headers
    dataf = data.values[1:, 1:]  # Skip the first row and column
    data = np.array(dataf)
    
    # Define features and targets for Model 2
    x = data[:, 18:619]  # Features from columns 18 to 618
    y = data[:, 619:711]  # Targets from columns 619 to 710
    
    # Split dataset into training and testing sets
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=point, random_state=42)
    
    # Initialize and fit the MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train_X)
    
    # Scale the datasets
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    x = scaler.transform(x)
    
    return train_X, test_X, train_y, test_y, x, y

# Objective function for Optuna optimization - Model 1
def objective1(trial):
    """
    Define the optimization objective function for Model 1 using Optuna.

    Parameters:
        trial (optuna.trial.Trial): An Optuna trial object to suggest hyperparameters.

    Returns:
        float: Mean validation loss (mean squared error) across cross-validation folds.
    """
    # Suggest hyperparameters from a log-uniform distribution
    c_value = trial.suggest_loguniform('c_value', 0.01, 100)
    length_scale = trial.suggest_loguniform('length_scale', 0.01, 100)
    
    # Create the first dataset
    train_X, test_X, train_y, test_y, x, y = create_dataset1(0.3)
    x_train = train_X.astype('float64')
    y_train = train_y.astype('float64')
    
    # Prepare K-Fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=random.randint(0, 100))
    val_losses = []

    # Loop over each fold
    for train_index, val_index in kf.split(x_train):
        # Split data into training and validation sets
        x_train_split, x_val_split = x_train[train_index], x_train[val_index]
        y_train_split, y_val_split = y_train[train_index], y_train[val_index]
    
        # Define and train the Gaussian Process Regressor
        kernel = C(c_value) * RBF(length_scale)
        model1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
        model1.fit(x_train_split, y_train_split)
    
        # Predict and calculate validation loss (MSE)
        y_val_pred = model1.predict(x_val_split)
        val_loss = mean_squared_error(y_val_split, y_val_pred)
    
        # Append the validation loss to the list
        val_losses.append(val_loss)
    
    return np.mean(val_losses)  # Return mean validation loss

# Objective function for Optuna optimization - Model 2
def objective2(trial):
    """
    Define the optimization objective function for Model 2 using Optuna.

    Parameters:
        trial (optuna.trial.Trial): An Optuna trial object to suggest hyperparameters.

    Returns:
        float: Mean validation loss (mean squared error) across cross-validation folds.
    """
    # Suggest hyperparameters from a log-uniform distribution
    c_value = trial.suggest_loguniform('c_value', 0.01, 100)
    length_scale = trial.suggest_loguniform('length_scale', 0.01, 100)
    
    # Create the second dataset
    train_X, test_X, train_y, test_y, x, y = create_dataset2(0.3)
    x_train = train_X.astype('float64')
    y_train = train_y.astype('float64')
    
    # Prepare K-Fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=random.randint(0, 100))
    val_losses = []

    # Loop over each fold
    for train_index, val_index in kf.split(x_train):
        # Split data into training and validation sets
        x_train_split, x_val_split = x_train[train_index], x_train[val_index]
        y_train_split, y_val_split = y_train[train_index], y_train[val_index]
    
        # Define and train the Gaussian Process Regressor
        kernel = C(c_value) * RBF(length_scale)
        model2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
        model2.fit(x_train_split, y_train_split)
    
        # Predict and calculate validation loss (MSE)
        y_val_pred = model2.predict(x_val_split)
        val_loss = mean_squared_error(y_val_split, y_val_pred)
    
        # Append the validation loss to the list
        val_losses.append(val_loss)
    
    return np.mean(val_losses)  # Return mean validation loss

# Use Optuna to optimize hyperparameters and visualize results
def optimize_and_visualize(objective_function, n_trials=20, direction='minimize'):
    """
    Perform hyperparameter optimization using Optuna and visualize the results.

    Parameters:
        objective_function (callable): The objective function for optimization.
        n_trials (int): Number of optimization trials. Default is 20.
        direction (str): Direction of optimization ('minimize' or 'maximize').

    Returns:
        dict: Best hyperparameters found during optimization.
    """
    with tqdm(total=n_trials, desc="Optimization Progress") as pbar:
        def callback(study, trial):
            pbar.update(1)  # Update progress bar for each trial

        # Create and optimize an Optuna study
        study = optuna.create_study(direction=direction)
        study.optimize(objective_function, n_trials=n_trials, callbacks=[callback])

    # Visualize optimization results
    optuna.visualization.plot_optimization_history(study)
    plt.show()

    optuna.visualization.plot_param_importances(study)
    plt.show()

    return study.best_params  # Return the best parameters found

# Function for retraining GP model with the best parameters
def retrain_gp_with_params(create_dataset_func, best_params):
    """
    Retrain a Gaussian Process model using the best hyperparameters obtained from Optuna optimization.

    Parameters:
        create_dataset_func (function): Function to create and prepare the dataset.
        best_params (dict): Best hyperparameters for model retraining.

    Returns:
        GaussianProcessRegressor: The trained Gaussian Process model.
    """
    # Create and prepare the dataset
    train_X, test_X, train_y, test_y, x, y = create_dataset_func(0.3)
    x_train = train_X.astype('float64')
    y_train = train_y.astype('float64')

    # Define and train Gaussian Process Regressor with best parameters
    kernel = C(best_params['c_value']) * RBF(best_params['length_scale'])
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)

    # Train the model
    model.fit(x_train, y_train)
    
    # Evaluate the model
    test_prediction = model.predict(test_X.astype('float64'))
    test_loss = mean_squared_error(test_y, test_prediction)
    print(f"Test Loss: {test_loss}")

    return model

# Class for handling cascaded models
class CascadedModel:
    def __init__(self, model1, model2):
        """
        Initialize a cascaded model by combining two models.

        Parameters:
            model1 (GaussianProcessRegressor): The first model in the cascade.
            model2 (GaussianProcessRegressor): The second model in the cascade.
        """
        self.model1 = model1
        self.model2 = model2

    def predict(self, X):
        """
        Perform predictions using the cascaded models.

        Parameters:
            X (np.ndarray): Input features for predictions.

        Returns:
            np.ndarray: The final predictions from the cascaded model.
        """
        # Perform prediction with the first model
        intermediate_prediction = self.model1.predict(X)
        
        # Use first model's output as input features for second model
        intermediate_prediction = intermediate_prediction.reshape(-1, 1)  # Ensure output is 2D
        final_prediction = self.model2.predict(intermediate_prediction)
        
        return final_prediction

    def save_model(self, filename):
        """
        Save the cascaded model to a file.

        Parameters:
            filename (str): The file path to save the model.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        """
        Load a cascaded model from a file.

        Parameters:
            filename (str): The file path from which to load the model.

        Returns:
            CascadedModel: The loaded cascaded model instance.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

# Main script execution
if __name__ == '__main__':
    # Optimize and find the best parameters using Optuna
    best_params1 = optimize_and_visualize(objective1, n_trials=20, direction='minimize')
    print("Best Parameters for Objective 1:", best_params1)
    
    best_params2 = optimize_and_visualize(objective2, n_trials=20, direction='minimize')
    print("Best Parameters for Objective 2:", best_params2)

    # Retrain models with the best parameters obtained
    model1 = retrain_gp_with_params(create_dataset1, best_params1)
    model2 = retrain_gp_with_params(create_dataset2, best_params2)

    # Create and save a cascaded model
    cascaded_model = CascadedModel(model1, model2)
    cascaded_model.save_model('cascaded_gp_model.pkl')

    # Load cascaded model from file and perform predictions
    loaded_cascaded_model = CascadedModel.load_model('cascaded_gp_model.pkl')
    _, test_X, _, test_y, _, _ = create_dataset1(0.3)
    test_X = test_X.astype('float64')

    predict = loaded_cascaded_model.predict(test_X)

    # Use Pandas ExcelWriter to save predictions
    with pd.ExcelWriter('predictions.xlsx') as writer:
        for i in range(predict.shape[0]):
            # Reshape predictions and actual values for output
            predict_subtable = predict[i].reshape(-1, 1)
            test_y_subtable = test_y[i].reshape(-1, 1)
        
            # Create DataFrame for the current prediction vs. actual values
            df = pd.DataFrame(np.hstack((predict_subtable, test_y_subtable)), columns=['Predict', 'Actual'])
        
            # Write DataFrame to a specific Excel sheet
            sheet_name = f'Subtable_{i+1}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Prediction results have been saved to 'predictions.xlsx'")



# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split
import optuna
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import random
import pickle

# Function to create the first dataset
def create_dataset1(point):
    """
    Create the first dataset by loading, processing, and splitting data from an Excel file.

    Parameters:
        point (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Scaled training and testing data (features and targets), and full dataset.
    """
    data = pd.read_excel('./data/data.xlsx', header=None)  # Load data from Excel file without headers
    dataf = data.values[1:, 1:]  # Skip the first row and column, if they are headers
    data = np.array(dataf)

    # Split data into features and targets
    x = data[:, 0:18]  # Features from columns 0 to 17
    y = data[:, 18:619]  # Targets from columns 18 to 618

    # Split into training and testing sets
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=point, random_state=42)

    # Initialize and fit the MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train_X)

    # Scale data
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    x = scaler.transform(x)

    return train_X, test_X, train_y, test_y, x, y


# Function to create the second dataset
def create_dataset2(point):
    """
    Create the second dataset by loading, processing, and splitting data from an Excel file.

    Parameters:
        point (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Scaled training and testing data (features and targets), and full dataset.
    """
    data = pd.read_excel('./data/data.xlsx', header=None)  # Load data from Excel file without headers
    dataf = data.values[1:, 1:]  # Skip the first row and column, if they are headers
    data = np.array(dataf)

    # Split data into features and targets
    x = data[:, 18:619]  # Features from columns 18 to 618
    y = data[:, 619:711]  # Targets from columns 619 to 710

    # Split into training and testing sets
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=point, random_state=42)

    # Initialize and fit the MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(train_X)

    # Scale data
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    x = scaler.transform(x)

    return train_X, test_X, train_y, test_y, x, y


# Objective function for Optuna optimization - Model 1
def objective1(trial):
    """
    Define the optimization objective function for Model 1 using Optuna.

    Parameters:
        trial (optuna.trial.Trial): An Optuna trial object to suggest hyperparameters.

    Returns:
        float: Mean validation loss (mean squared error) across cross-validation folds.
    """
    # Suggest hyperparameters from a log-uniform distribution
    svc_c = trial.suggest_loguniform('svc_c', 0.00001, 0.01)
    svc_gamma = trial.suggest_loguniform('svc_gamma', 0.00001, 0.1)

    # Create first dataset
    train_X, test_X, train_y, test_y, x, y = create_dataset1(0.3)
    x_train = train_X.astype('float64')
    y_train = train_y.astype('float64')

    # Prepare K-Fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=random.randint(0, 100))
    val_losses = []

    # Loop over each fold
    for train_index, val_index in kf.split(x_train):
        # Split data into training and validation sets
        x_train_split, x_val_split = x_train[train_index], x_train[val_index]
        y_train_split, y_val_split = y_train[train_index], y_train[val_index]

        # Define and train the MultiOutputRegressor with SVR
        model1 = MultiOutputRegressor(SVR(C=svc_c, gamma=svc_gamma))
        model1.fit(x_train_split, y_train_split)

        # Predict and calculate validation loss (MSE)
        y_val_pred = model1.predict(x_val_split)
        val_loss = mean_squared_error(y_val_split, y_val_pred)

        # Append the validation loss to the list
        val_losses.append(val_loss)

    return np.mean(val_losses)  # Return mean validation loss


# Objective function for Optuna optimization - Model 2
def objective2(trial):
    """
    Define the optimization objective function for Model 2 using Optuna.

    Parameters:
        trial (optuna.trial.Trial): An Optuna trial object to suggest hyperparameters.

    Returns:
        float: Mean validation loss (mean squared error) across cross-validation folds.
    """
    # Suggest hyperparameters from a log-uniform distribution
    svc_c = trial.suggest_loguniform('svc_c', 0.00001, 0.01)
    svc_gamma = trial.suggest_loguniform('svc_gamma', 0.00001, 0.1)

    # Create second dataset
    train_X, test_X, train_y, test_y, x, y = create_dataset2(0.3)
    x_train = train_X.astype('float64')
    y_train = train_y.astype('float64')

    # Prepare K-Fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=random.randint(0, 100))
    val_losses = []

    # Loop over each fold
    for train_index, val_index in kf.split(x_train):
        # Split data into training and validation sets
        x_train_split, x_val_split = x_train[train_index], x_train[val_index]
        y_train_split, y_val_split = y_train[train_index], y_train[val_index]

        # Define and train the MultiOutputRegressor with SVR
        model2 = MultiOutputRegressor(SVR(C=svc_c, gamma=svc_gamma))
        model2.fit(x_train_split, y_train_split)

        # Predict and calculate validation loss (MSE)
        y_val_pred = model2.predict(x_val_split)
        val_loss = mean_squared_error(y_val_split, y_val_pred)

        # Append the validation loss to the list
        val_losses.append(val_loss)

    return np.mean(val_losses)  # Return mean validation loss


# Optimize hyperparameters using Optuna and visualize results
def optimize_and_visualize(objective_function, n_trials=20, direction='minimize'):
    """
    Perform hyperparameter optimization using Optuna and visualize the results.

    Parameters:
        objective_function (callable): The objective function for optimization.
        n_trials (int): Number of optimization trials. Default is 20.
        direction (str): Direction of optimization ('minimize' or 'maximize').

    Returns:
        dict: Best hyperparameters found during optimization.
    """
    with tqdm(total=n_trials, desc="Optimization Progress") as pbar:
        def callback(study, trial):
            pbar.update(1)  # Update the progress bar for each trial

        # Create and optimize an Optuna study
        study = optuna.create_study(direction=direction)
        study.optimize(objective_function, n_trials=n_trials, callbacks=[callback])

    # Visualize optimization results
    optuna.visualization.plot_optimization_history(study)
    plt.show()

    optuna.visualization.plot_param_importances(study)
    plt.show()

    return study.best_params  # Return the best hyperparameters found


# Function to retrain models with optimal parameters (this function is defined with an incorrect name, and needs actual implementation based on model)
def retrain_svr_with_params(create_dataset_func, best_params):
    """
    Retrain SVR model using the best hyperparameters obtained from Optuna optimization.

    Parameters:
        create_dataset_func (function): Function to create and prepare the dataset.
        best_params (dict): Best hyperparameters for model retraining.

    Returns:
        MultiOutputRegressor: The trained multi-output SVR model.
    """
    # Create and prepare the dataset
    train_X, test_X, train_y, test_y, x, y = create_dataset_func(0.3)
    x_train = train_X.astype('float64')
    y_train = train_y.astype('float64')

    # Initialize SVR model with best hyperparameters
    model = MultiOutputRegressor(SVR(C=best_params['svc_c'], gamma=best_params['svc_gamma']))
    model.fit(x_train, y_train)  # Train the model

    # Evaluate the model
    test_prediction = model.predict(test_X.astype('float64'))
    test_loss = mean_squared_error(test_y, test_prediction)
    print(f"Test Loss: {test_loss}")

    return model  # Return the trained model


# Class for handling cascaded models
class CascadedModel:
    def __init__(self, model1, model2):
        """
        Initialize a cascaded model by combining two models.

        Parameters:
            model1: The first model in the cascade.
            model2: The second model in the cascade.
        """
        self.model1 = model1
        self.model2 = model2

    def predict(self, X):
        """
        Perform predictions using the cascaded models.

        Parameters:
            X (np.ndarray): Input features for predictions.

        Returns:
            np.ndarray: The final predictions from the cascaded model.
        """
        # Perform prediction with the first model
        intermediate_prediction = self.model1.predict(X)

        # Use first model's output as input features for second model
        intermediate_prediction = intermediate_prediction.reshape(-1, 1)  # Ensure output is 2D
        final_prediction = self.model2.predict(intermediate_prediction)

        return final_prediction

    def save_model(self, filename):
        """
        Save the cascaded model to a file.

        Parameters:
            filename (str): The file path to save the model.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        """
        Load a cascaded model from a file.

        Parameters:
            filename (str): The file path from which to load the model.

        Returns:
            CascadedModel: The loaded cascaded model instance.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)


# Main script execution
if __name__ == '__main__':
    # Optimize and find the best parameters using Optuna
    best_params1 = optimize_and_visualize(objective1, n_trials=20, direction='minimize')
    print("Best Parameters for Objective 1:", best_params1)
    
    best_params2 = optimize_and_visualize(objective2, n_trials=20, direction='minimize')
    print("Best Parameters for Objective 2:", best_params2)

    # Retrain models with the obtained optimal hyperparameters
    model1 = retrain_svr_with_params(create_dataset1, best_params1)
    model2 = retrain_svr_with_params(create_dataset2, best_params2)

    # Create and save a cascaded model
    cascaded_model = CascadedModel(model1, model2)
    cascaded_model.save_model('cascaded_svr_model.pkl')

    # Load cascaded model from file and perform predictions
    loaded_cascaded_model = CascadedModel.load_model('cascaded_svr_model.pkl')
    _, test_X, _, test_y, _, _ = create_dataset1(0.3)
    test_X = test_X.astype('float64')

    predict = loaded_cascaded_model.predict(test_X)

    # Use Pandas ExcelWriter to save predictions
    with pd.ExcelWriter('predictions.xlsx') as writer:
        for i in range(predict.shape[0]):
            # Reshape predictions and actual values for output
            predict_subtable = predict[i].reshape(-1, 1)
            test_y_subtable = test_y[i].reshape(-1, 1)

            # Create DataFrame for the current prediction vs. actual values
            df = pd.DataFrame(np.hstack((predict_subtable, test_y_subtable)), columns=['Predict', 'Actual'])
            
            # Write DataFrame to a specific Excel sheet
            sheet_name = f'Subtable_{i+1}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Prediction results have been saved to 'predictions.xlsx'")



#%% pdp
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import shap
from models import ProbAttention
from keras.models import load_model
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Function to create dataset and split it into training and testing sets
def create_dataset1(point):
    """
    Create the first dataset by loading data from an Excel file, processing it, and splitting into train and test sets.

    Parameters:
        point (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: Scaled training and testing data (features and targets), and full dataset.
    """
    # Load data from Excel file without headers
    data = pd.read_excel('./data/data.xlsx', header=None)
    dataf = data.values[1:, 1:]  # Skip the first row and column for data
    data = np.array(dataf)

    # Split data into features and targets
    x = data[:, 0:18]  # Select features from columns 0 to 17
    y = data[:, 18:619]  # Select targets from columns 18 to 618

    # Split the dataset into training and testing sets with specified test size
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=point, random_state=42)

    # Initialize and fit the MinMaxScaler on the training features, then transform both train and test sets
    scaler = MinMaxScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    
    # Return all the related datasets
    x = scaler.transform(x)
    return train_X, test_X, train_y, test_y, x, y

# Load the entire dataset and extract feature names for SHAP analysis
data = pd.read_excel('./data/data.xlsx')
column_names = data.columns  # Fetch column names for later use in SHAP analysis
selected_column_names = column_names[1:19]  # Select column names corresponding to features

dataf = data.values
dataf = dataf[1:, 1:]  # Drop header row and the first column assuming non-data purpose

data = np.array(dataf)
x = data[:, 0:18]  # Extract feature set

# Load the pre-trained model
# `ProbAttention` is assumed as a custom object used during model creation
model = load_model('combined_model.h5', custom_objects={'ProbAttention': ProbAttention})

# Create SHAP Explainer using the model and the feature set `x`
explainer = shap.Explainer(model, x)

# Compute SHAP values for the dataset `x`
shap_values = explainer(x)

# Save SHAP values to an Excel file for analysis
df = pd.DataFrame(shap_values.values, columns=selected_column_names)
filename = 'shap_values.xlsx'
df.to_excel(filename, index=False)

# Code concludes here with 'shap_values.xlsx' containing the relevance of each feature
# towards the predictions made by the loaded Attention model




#%% pdp
from econml.solutions.causal_analysis import CausalAnalysis
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# Define feature names to be used from the dataset
feature_names = [
    'SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 
    'B1_1', 'B2_1', 'B3_1', 'B4_1', 'B5_1', 'B6_1', 'B7_1', 'B8A_1', 
    'B8_1', 'B11_1', 'B12_1'
]

# Create an ExcelWriter object to save output results
writer1 = pd.ExcelWriter('Causal_relationship_1.xlsx')

# Read labels from an Excel file, apply log transformation to labels for stabilization
Labels = pd.read_excel('./data/data1.xlsx').iloc[:, 18:]
Labels = np.log(Labels + 1)
Label_names = Labels.columns  # Store label names for use in Excel sheet naming
print(Labels)

# Loop over the first five labels to analyze causal effects
for j in range(5):
    print(j)

    # Extract the current label as the target variable
    y = pd.DataFrame(Labels.iloc[:, j])
    l_names = y.columns.values.tolist()  # Get column names for the label

    # Load feature data from Excel, selecting only specified features
    Features = pd.read_excel('./data/data1.xlsx')
    Features = Features[feature_names]
    print(Features)

    x = Features.copy()  # Copy features for processing
    f_names = x.columns  # Get column names for features

    # Standardize feature data
    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(x), columns=f_names)

    # Initialize CausalAnalysis object with specified parameters
    ca = CausalAnalysis(
        feature_inds=f_names,
        categorical=[],  # Specify categorical indicators, if any
        classification=False,  # Indicates a regression task (continuous target variable)
        nuisance_models='automl',  # Use automatic machine learning for nuisance models
        heterogeneity_model="forest",  # Use forest model for heterogeneity
        mc_iters=20,  # Number of Monte Carlo iterations
        random_state=123,  # Seed for reproducibility
        n_jobs=-1  # Enable parallel processing with all available cores
    )
    ca.fit(x, np.ravel(y))  # Fit the causal analysis model

    # Calculate global causal effects and sort by p-value
    global_summ = ca.global_causal_effect(alpha=0.05)
    global_summ = global_summ.sort_values(by="p_value")

    # Write the global causal effects to the Excel file
    global_summ.to_excel(writer1, sheet_name=l_names[0])

# Close the ExcelWriter to finalize the first set of results
writer1.close()

'''
-------------------------- Causal relationship among features ---------------------------
'''

from econml.solutions.causal_analysis import CausalAnalysis
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# Initialize a DataFrame to store results
result = pd.DataFrame()

# Loop for a single run - placeholder as template expansion
for j in range(1):
    
    # Load feature data from Excel, selecting only specified features
    Features = pd.read_excel('./data/data1.xlsx')
    Features = Features[feature_names]
    
    x = Features.copy()  # Copy features for processing
    f_names = x.columns  # Get column names for features

    # Standardize feature data
    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(x), columns=f_names)
    n_f = len(f_names)  # Get the number of features

    # Analyze causal relationship for each feature as the target
    for i in range(n_f):
        # Define current target feature and drop it from the predictors
        x = Features.drop(f_names[i], axis=1)
        y = pd.DataFrame(Features.iloc[:, i])
        print(y.columns[0])

        # Initialize CausalAnalysis object with specified parameters
        ca = CausalAnalysis(
            feature_inds=x.columns,
            categorical=[],  # Specify categorical indicators, if any
            classification=False,  # Indicates a regression task
            nuisance_models="automl",  # Use automated machine learning
            heterogeneity_model="forest",  # Use forest model for heterogeneity
            mc_iters=20,  # Monte Carlo iterations
            n_jobs=-1,  # Utilize all available cores for parallel processing
            random_state=123
        )
        ca.fit(x, np.ravel(y))  # Fit the model to data

        # Calculate global causal effects and sort by p-value
        global_summ = ca.global_causal_effect(alpha=0.05)
        global_summ = global_summ.sort_values(by="p_value")

        # Add additional columns for context and concatenate results
        global_summ.insert(0, 'Function', 'pollution')
        global_summ.insert(0, 'label_feature', y.columns[0])

        # Concatenate current iteration's results to the main result DataFrame
        result = pd.concat([result, global_summ])

# Save the aggregated causal relationship results to another Excel file
result.to_excel('Causal_relationship_2.xlsx')


#%% pdp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from patsy import dmatrix
from statsmodels.gam.api import GLMGam, BSplines
import os

# Configure matplotlib for displaying plots in a way that ensures compatibility with different fonts
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Set default font to 'Microsoft YaHei' to support Chinese characters
plt.rcParams['axes.unicode_minus'] = False  # Correctly display minus signs on plots

# Initialize a DataFrame to store R-squared and p-values for each feature's analysis
r2_p = pd.DataFrame(columns=['feature', 'r2', 'p'])

# Load shap_values and features data from Excel files
shap_values = pd.read_excel('shap_values.xlsx')
features = pd.read_excel('Features selected.xlsx').iloc[:, :]
columns = pd.read_excel('Features selected.xlsx').columns  # Load column names for feature references

# Use an Excel writer object to save resulting plots and analysis to an Excel file
with pd.ExcelWriter('shap_analysis_results.xlsx', engine='xlsxwriter') as writer:
    # Loop through the first seven features for SHAP value analysis
    for j in range(7):
        # Prepare data for current iteration containing feature and corresponding SHAP values
        data = pd.DataFrame({
            'variable': features.iloc[:, j],  # Feature values
            'shap_values': shap_values.iloc[:, j]  # Corresponding SHAP values
        })
        
        # Retrieve the name of the current feature for labeling the plot and sheet name
        f_name = columns[columns.index(shap_values.columns[j])]

        # Standardize the feature data for better modeling and outlier detection
        temp_data = (data['variable'] - np.mean(data['variable'])) / np.std(data['variable'])
        temp_index = temp_data[(temp_data > 3) | (temp_data < -3)].index  # Detect outliers beyond 3 standard deviations
        data = data.drop(temp_index)  # Remove detected outliers from the dataset

        # Generate B-spline basis functions for generalized additive model (GAM)
        x_spline = dmatrix("bs(variable, df=4)", {"variable": data['variable']}, return_type='dataframe')

        # Create the GAM model using B-splines for smoothing and fitting to SHAP values
        gam = GLMGam.from_formula(
            'shap_values ~ x_spline - 1', 
            data=data, 
            smoother=BSplines(data['variable'], df=[4], degree=[3])  # Define smoother parameters
        )
        
        # Fit the GAM model and compute R-squared value
        result = gam.fit()
        ss_total = np.sum((data['shap_values'] - np.mean(data['shap_values']))**2)  # Total variance in the data
        ss_residual = np.sum((data['shap_values'] - result.fittedvalues)**2)  # Model residuals
        r2 = 1 - ss_residual / ss_total  # Compute R-squared value indicating fit quality
        
        # Save model p-value and R-squared in a DataFrame
        p = result.pvalues[0]
        temp = pd.DataFrame({'feature': [shap_values.columns[j]], 'r2': [r2], 'p': [p]})
        r2_p = pd.concat([r2_p, temp], ignore_index=True)
        
        # Extract fitted values and standard errors for visualization
        standard_errors = result.bse
        predictions = result.get_prediction()
        standard_errors = predictions.se_mean  # Obtain standard error of the mean for predictions

        # Create and configure scatterplot and lineplot for SHAP analysis visualization
        plt.figure(figsize=(5, 3.5))
        sns.scatterplot(x='variable', y='shap_values', data=data, color='#0089FA', s=20)
        sns.lineplot(x=data['variable'], y=result.fittedvalues, color='red')

        # Annotate plot with R-squared and p-value
        x_range = data['variable'].max() - data['variable'].min()
        y_range = data['shap_values'].max() - data['shap_values'].min()
        x_position = data['variable'].min() + 0.05 * x_range
        y_position_r2 = data['shap_values'].max() - 0.05 * y_range
        y_position_p = data['shap_values'].max() - 0.15 * y_range

        plt.annotate(f'$R^2 = {r2:.3f}$', xy=(x_position, y_position_r2), fontsize=10, ha='left', va='baseline')
        plt.annotate(f'$p < 0.05$', xy=(x_position, y_position_p), fontsize=10, ha='left', va='baseline')
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)  # Add horizontal line at y=0

        plt.xlabel(f_name)  # Label x-axis with feature name
        plt.ylabel('Shap value')  # Label y-axis with metric name
        plt.title(f'SHAP Analysis for {f_name}')  # Set plot title
        plt.grid(False)
        plt.tight_layout()  # Adjust layout to prevent clipping

        # Define save path and directory for plots
        save_dir = './'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # Create directory if it doesn't exist

        save_path = os.path.join(save_dir, f'{f_name}_shap_analysis.png')
        plt.savefig(save_path, dpi=100)  # Save the plot as a PNG file
        plt.close()  # Close the plot to free memory

        # Save the variable and fitted values for each feature's analysis to a separate sheet
        output_df = pd.DataFrame({
            'variable': data['variable'], 
            'fitted_values': result.fittedvalues
        })
        output_df.to_excel(writer, sheet_name=f_name, index=False)

# Export the collected R-squared and p-values for each feature to an Excel file
r2_p.to_excel('r2_p_results.xlsx', index=False)
