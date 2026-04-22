import numpy as np
import pandas as pd
import h5py
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



def label_data(input, output, label):

    df = pd.read_csv(input, on_bad_lines='skip')

    df['label'] = label

    df.to_csv(output, index = False)

def combine_data(dataframes, output):

    new_df = pd.concat(dataframes, ignore_index=True)
    new_df.to_csv(output, index = False)

def segment_and_shuffle_data(input, output):
    df = pd.read_csv(input, on_bad_lines='skip')
    segments = []
    for i in range(0, len(df), 500): # Iterates through dataframe in step size 500 (assuming 0.01 seconds per datapoint so 5*100 = 500 per 5 seconds)
        segment = df.iloc[i:i+500]
        if len(segment) == 500:
            segments.append(segment)

    shuffled_segments = shuffle(segments) # Shuffle list
    shuffled_data = pd.concat(shuffled_segments,ignore_index=True) # Concatenate back to Dataframe
    shuffled_data.to_csv(output, index=False)

    print(shuffled_data)

def split(input, testFile, trainFile):
    df = pd.read_csv(input, on_bad_lines='skip')
    train, test = train_test_split(df, test_size = 0.1)

    train.to_csv(testFile, index=False)
    test.to_csv(trainFile, index = False)




def store_data():
    with h5py.File('./data.hdf5', 'w') as hdf:
        for member in ['josh', 'charlie', 'maddy']:
            member_group = hdf.create_group(member)

            for activity in ['walking', 'jumping']:
                file_path = f'C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\{member}_{activity}_shuffled.csv'
                data = pd.read_csv(file_path).to_numpy()
                member_group.create_dataset(f'{activity}', data=data)

        for activity in ['walking', 'jumping']:
            train_file_path = f'C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\{activity}_training.csv'
            train_data = pd.read_csv(train_file_path).to_numpy()
            hdf.create_dataset(f'{activity}_train', data=train_data)

            test_file_path = f'C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\{activity}_testing.csv'
            test_data = pd.read_csv(test_file_path).to_numpy()
            hdf.create_dataset(f'{activity}_test', data=test_data)



# Josh Data
label_data('C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\josh_walking.csv','C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\josh_walking_labelled.csv', 0)
label_data('C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\josh_jumping.csv','C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\josh_jumping_labelled.csv', 1)
# Charlie Data
label_data('C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\charlie_walking.csv','C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\charlie_walking_labelled.csv', 0)
label_data('C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\charlie_jumping.csv','C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\charlie_jumping_labelled.csv', 1)
# Maddy Data
label_data('C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\maddy_walking.csv','C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\maddy_walking_labelled.csv', 0)
label_data('C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\maddy_jumping.csv','C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\maddy_jumping_labelled.csv', 1)




segment_and_shuffle_data('C:\\Users\\joshp\PycharmProjects\Finalproject292hdf5\\josh_jumping_labelled.csv', 'C:\\Users\\joshp\PycharmProjects\Finalproject292hdf5\\josh_jumping_shuffled.csv')
segment_and_shuffle_data('C:\\Users\\joshp\PycharmProjects\Finalproject292hdf5\\josh_walking_labelled.csv', 'C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\josh_walking_shuffled.csv')
segment_and_shuffle_data('C:\\Users\\joshp\PycharmProjects\Finalproject292hdf5\\charlie_jumping_labelled.csv', 'C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\charlie_jumping_shuffled.csv')
segment_and_shuffle_data('C:\\Users\\joshp\PycharmProjects\Finalproject292hdf5\\charlie_walking_labelled.csv', 'C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\charlie_walking_shuffled.csv')
segment_and_shuffle_data('C:\\Users\\joshp\PycharmProjects\Finalproject292hdf5\\maddy_jumping_labelled.csv', 'C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\maddy_jumping_shuffled.csv')
segment_and_shuffle_data('C:\\Users\\joshp\PycharmProjects\Finalproject292hdf5\\maddy_walking_labelled.csv', 'C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\maddy_walking_shuffled.csv')

walking_dataframes = [
    pd.read_csv('C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\josh_walking_labelled.csv'),
    pd.read_csv('C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\charlie_walking_labelled.csv'),
    pd.read_csv('C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\maddy_walking_labelled.csv')
]

jumping_dataframes = [
    pd.read_csv('C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\josh_jumping_labelled.csv'),
    pd.read_csv('C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\charlie_jumping_labelled.csv'),
    pd.read_csv('C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\maddy_jumping_labelled.csv')
]

combine_data(walking_dataframes, 'C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\walking_data.csv' )
combine_data(jumping_dataframes, 'C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\jumping_data.csv' )
split('C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\walking_data.csv', 'C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\walking_testing.csv', 'C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\walking_training.csv')
split('C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\jumping_data.csv', 'C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\jumping_testing.csv', 'C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\jumping_training.csv')

walking_training_df = pd.read_csv('C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\walking_training.csv')
jumping_training_df = pd.read_csv('C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\jumping_training.csv')

walking_testing_df = pd.read_csv('C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\walking_testing.csv')
jumping_testing_df = pd.read_csv('C:\\Users\\joshp\\PycharmProjects\\Finalproject292hdf5\\jumping_testing.csv')
store_data()

print(walking_training_df.head())
