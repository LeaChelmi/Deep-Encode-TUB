import os

def test():
    test = input('prompt\n')
    if test == 'yes':
        os.system("python3 test_call.py")


def main():

    print('HELLOOO')
    print(os.path.dirname)
    print(os.getcwd())
    if os.path.isdir('misc'):
        print('FOUND MISC')
        print(os.listdir('misc'))
    

    # evaluation or predictions
    eval = ''
    while (eval != 'e' and eval != 'p'):
        eval = input('----------------------------------\nHello!\nThis is the video encoding prediction tool of Deep Encode Group 2. Do you want to test the pipeline, run training, test and Evaluation or just use the Prediction fucntionality? [e / p]\n----------------------------------\n')

    # choose learning model
    learning_model = ''
    while (learning_model != 'nn' and learning_model != 'dt'):
        learning_model = input('----------------------------------\nThis tool features a Neural Network and a Decision Tree Classifier to predict optimal bitrate achieving an arbitrary VMAF value.\nIn a first step, please enter the Learning Model you want to use [nn / dt]\n----------------------------------\n')
    
    if (eval == 'e'):
        eval_run(learning_model)
    elif (eval == 'p'):
        pred_run(learning_model)
    else:
        print('ERROR! No mode selected')
        
        
    
        
def eval_run(learning_model):
    # new training data
    new_data = ''
    while (new_data != 'y' and new_data != 'n'):
        new_data = input('----------------------------------\nDo you want to add data to the data set before running training, testing and evalutaion? [y / n]\n----------------------------------\n')
    
    # select whether NN params should be adjusted accordingly, takes very long
    if (learning_model == 'nn' and new_data == 'y'):
        adjust_nn = ''
        while (adjust_nn != 'y' and adjust_nn != 'n'):
            adjust_nn = input('----------------------------------\nThe Neural Network has been designed in a way that reflects our training data set. It can be adapted to refelct the new training data set. However, this can take a while! Do you want to adjust the NN? [y / n]\n----------------------------------\n')

    # if new data should be added to data set, trigger labeling pipeline
    if (new_data == 'y'):

        # get data set path
        dataset_path = os.path.join(os.getcwd(), 'default_dataset')

        # run metadata extraction
        os.system('python3 extract_metadat.py')

        # run create_folders.py
        os.system('python3 create_folders.py')

        # run split_scenes.py
        os.system('python3 split_scenes.py')

        # labeling
        os.system('python3 labeling.py')

        # feature extraction
        os.system('python3 feature_extraction.py')

        # merge features and labels of new data
        os.system('python3 merge_feature_labels.py')

        # merge new data to existing dataset 
        #TODO

    if (learning_model == 'dt'):
        os.system('python3 DT_train_and_test.py')
    
    
    elif (learning_model == 'nn'):
        if (adjust_nn == 'y'):
            #TODO
            print('smth')
        #TODO
        print('smth')
    else:
        print('ERROR')
    
    # evaluate
    os.system('python3 eval.py')



def pred_run(learning_model):
    
    # run metadata extraction
    os.system('python3 extract_metadat.py')

    # run create_folders.py
    os.system('python3 create_folders.py')

    # run split_scenes.py
    os.system('python3 split_scenes.py')

    # feature extraction
    os.system('python3 feature_extraction.py')

    # 
    if (learning_model == 'dt'):
        os.system('python3 DT_prediction.py')
    elif (learning_model == 'nn'):
        print('TODO')
    else:
        print('ERROR')


if __name__ == "__main__":
    main()