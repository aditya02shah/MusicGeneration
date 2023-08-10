

# Music Generation Model
This repository contains a music generation model that generates original music compositions using deep learning techniques and TensorFlow

# Dataset
The music generation model is trained on the ABC notation dataset provided in the MIT Deep Learning Library(MDL).

The dataset contains 817 songs in text format. 
These songs will be used to train the model to generate new music.

# Introduction
ABC files contain information about the notes, timing, and velocity of each sound, making them well-suited for music generation using deep learning techniques.

By training the model on a dataset of existing songs, the model can learn patterns and structures in music and generate new melodies, harmonies, and rhythms. 

LSTM model can generate music that is both pleasing to the ear and follows a coherent musical structure. 



# Model Architecture
The model architecture is based on a recurrent neural network (RNN) with LSTM cells.
 The LSTM cells allow the model to capture long-term dependencies and generate coherent and melodic musical sequences.



# Generation
To generate new music using the trained model, run the generate.py script.  
Specify the desired length of the generated sequence and the output file path. The script will utilize the trained model to create an original composition.

# Results
Some sample compositions generated by the model have been included in the results directory.

