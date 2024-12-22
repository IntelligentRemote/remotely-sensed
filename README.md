Firstly, make sure Anaconda is installed on your system. Open a terminal or command prompt and use the following command to create a new virtual environment named myenv with a specified Python version (e.g., 3.8):

conda create --name myenv python=3.8
Once the environment is created, activate it using:

conda activate myenv
With the virtual environment activated, install all the dependencies listed in the requirements.txt file (assuming the file is located in the current directory) using the pip command:

pip install -r requirements.txt

Finally, to run the main.py file located in your project folder, use the following command while your virtual environment is still active:

python main.py
This will execute the main.py script using the Python interpreter within the activated virtual environment, ensuring all necessary dependencies and configurations are correctly applied.
