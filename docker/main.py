import os
import extract_metadata
import create_folders
import split_scenes
import labeling
import feature_extraction
import merge_features_labels
import merge_to_final_dataset
import DT_train_test
import NN_train_test
import time
import dt_prediction
import NN_prediction


def test():
    test = input('prompt\n')
    if test == 'yes':
        os.system("python3 test_call.py")


def main():

    #print('HELLOOO')
    #print(os.path.dirname)
    #print(os.getcwd())
    #if os.path.isdir('misc'):
    #    print('FOUND MISC')
    #    print(os.listdir('misc'))
    

    # evaluation or predictions
    eval = ''
    while (eval != 'e' and eval != 'p'):
        eval = input('----------------------------------\nHello!\nThis is the video encoding prediction tool of Deep Encode Group 2. Do you want to test the labeling pipeline, run training & test (use case 1, tt) or use the Prediction fucntionality? [tt / p]\n----------------------------------\n')

    # choose learning model
    learning_model = ''
    while (learning_model != 'nn' and learning_model != 'dt' and learning_model != 'both'):
        learning_model = input('----------------------------------\nThis tool features a Neural Network and a Decision Tree Classifier to predict optimal bitrate achieving an arbitrary VMAF value.\nIn a first step, please enter the Learning Model you want to use or both [nn / dt / both]\n----------------------------------\n')
    
    if (eval == 'e'):
        eval_run(learning_model)
    elif (eval == 'p'):
        pred_run(learning_model)
    else:
        print('ERROR! No mode selected')
        
        
    
        
def eval_run(learning_model):

     # get data set path
    dataset_path = os.path.join(os.getcwd(), 'videos_to_add_to_dataset')

    # new training data
    new_data = ''
    while (new_data != 'y' and new_data != 'n'):
        new_data = input('----------------------------------\nDo you want to add data to the data set before running training, testing and evalutaion? [y / n]\n----------------------------------\n')
        print('----------------------------------\n')

    # if new data should be added to data set, trigger labeling pipeline
    if (new_data == 'y'):

        # run metadata extraction
        print('Running Metadata Extraction...')
        extract_metadata.createMetadataCsv(dataset_path)
        print('\n----------------------------------\n')

        # run create_folders.py
        print('Creating Folders...')
        create_folders.organize_files(dataset_path)
        print('\n----------------------------------\n')

        # run split_scenes.py
        print('Splitting Scenes...')
        split_scenes.split_scenes(dataset_path)
        print('\n----------------------------------\n')

        # labeling
        print('Running Labeling...')
        labeling.labeling(dataset_path)
        print('\n----------------------------------\n')

        # feature extraction
        print('Running Feature Extraction...')
        feature_extraction.CreateFeatureTable(dataset_path)
        print('\n----------------------------------\n')

        # merge features and labels of new data
        print('Merging Features and Labels')
        merge_features_labels.mergeFeaturesAndLabels(dataset_path)
        print('\n----------------------------------\n')

        time.sleep(1)

        # merge new data to existing dataset 
        print('Merging new data to existing dataset')
        merge_to_final_dataset.merge_to_final_dataset(dataset_path)
        print('\n----------------------------------\n')

        time.sleep(1)



    if (learning_model == 'dt' or learning_model == 'both'):
        print('Running Descision Tree Training and Test...\n')
        DT_train_test.runDecisionTreeRegressor(dataset_path)
        print('\n----------------------------------\n')
        # print('Running Evaluation...')
        # eval.run_evaluation(dataset_path, 'DT')
        # print('----------------------------------\n')
    
    if (learning_model == 'nn' or learning_model == 'both'):
        print('Running Neural Network Training and Test...\n')
        NN_train_test.runNeuralNetwork(dataset_path)
        print('\n----------------------------------\n')
        # print('Running Evaluation...')
        # eval.run_evaluation(dataset_path, 'NN')
        # print('----------------------------------\n')
    
    


def pred_run(learning_model):

    # get data set path
    dataset_path = os.path.join(os.getcwd(), 'videos_to_predict')
    
    # run metadata extraction
    print('----------------------------------\n')
    print('Running Metadata Extraction...')
    extract_metadata.createMetadataCsv(dataset_path)
    print('\n----------------------------------\n')

    # run create_folders.py
    print('Creating Folders...')
    create_folders.organize_files(dataset_path)
    print('\n----------------------------------\n')

    # run split_scenes.py
    print('Splitting Scenes...')
    split_scenes.split_scenes(dataset_path)
    print('\n----------------------------------\n')

    # feature extraction
    print('Running Feature Extraction...')
    feature_extraction.CreateFeatureTable(dataset_path)
    print('\n----------------------------------\n')

    time.sleep(1)

    if (learning_model == 'dt' or learning_model == 'both'):
        print('Running Descision Tree Prediction...\n')
        dt_prediction.runPredictionBasedOnOurDataSet(dataset_path)
        print('\n----------------------------------\n')
    
    if (learning_model == 'nn' or learning_model == 'both'):
        print('Running Neural Network Prediction...\n')
        NN_prediction.runPredictionBasedOnOurDataSetNN(dataset_path)
        print('\n----------------------------------\n')


if __name__ == "__main__":
    main()