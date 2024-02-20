from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, cohen_kappa_score
from skmultiflow.data import DataStream
from sklearn.dummy import DummyClassifier
import math
import warnings
import argparse
import os

from model import Model
import random
from typing import List
from skmultiflow.data import DataStream
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from debugger import Debugger
from parameters import Parameters

def uniqueElement(arr, n):
    # Create a set
    s = set(arr)     
    # Compare and print the result
    if(len(s) == 1):
        return True
    else:
        return False
    
def Fmeasure(y_true,y_pred):
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true,y_pred, average='weighted')
    return macro_f1,weighted_f1

def kappa_score(y_true,y_predict):
    if uniqueElement(y_true,len(y_true)):
        return 1
    else:
        kappa_coefficient = cohen_kappa_score(y_true, y_predict)
        return kappa_coefficient
#p0: accuracy
#pe: accuracy of no-change classifier(dummy)
def kappa_plus_score(p0,pe):
    if pe == 1 and p0 == 1:
        return 0
    return(p0-pe)/(1-pe)

def kappa_PP_score(p0,pe):
    return (p0-pe)/math.sqrt(1+pow((1-pe),2))

def dummy_classifier(X_train,y_train):
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    dummy_clf.predict(X_train)
    pe = dummy_clf.score(X_train, y_train)
    return pe

def ReadISOT():
    df = pd.read_csv("./Datasets/shuffled_ISOT.csv")
    X_train = df.drop("y", axis=1).values
    y_train = df["y"].values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    return X_train,y_train

def load_data(packet_num=1):
    fpath = '../PyTPG/Datasets/CTU-13-Dataset/'+str(packet_num)+'/' + \
                    [fname 
                    for fname in os.listdir('../PyTPG/Datasets/CTU-13-Dataset/'+str(packet_num)) 
                        if fname.find('.binetflow') > 0][0]
    
    data = pd.read_csv(fpath)
    sub_data = data[['Dur','Proto','Dir','dTos','sTos','TotPkts','TotBytes','SrcBytes','Label']]

    sub_data['Dir'] = sub_data['Dir'].replace('  <->',0)
    sub_data['Dir'] = sub_data['Dir'].replace('  <?>',1)
    sub_data['Dir'] = sub_data['Dir'].replace('   ->',2)
    sub_data['Dir'] = sub_data['Dir'].replace('   ?>',3)
    sub_data['Dir'] = sub_data['Dir'].replace('  <-',-1)
    sub_data['Dir'] = sub_data['Dir'].replace('  <?',-2)
    sub_data['Dir'] = sub_data['Dir'].replace('  who',4)

    sub_data['dTos'] = sub_data['dTos'].replace(np.nan,-1)
    sub_data['sTos'] = sub_data['sTos'].replace(np.nan,-1)

    proto = sub_data['Proto'].unique()
    for i,p in enumerate(proto):
        sub_data['Proto'] = sub_data['Proto'].replace(p,i)

    Background = sub_data[sub_data['Label'].str.contains('Background')]
    Background['Label'] = 2
    Normal = sub_data[sub_data['Label'].str.contains('Normal')]
    Normal['Label'] = 0
    Botnet = sub_data[sub_data['Label'].str.contains('Botnet')]
    Botnet['Label'] = 1

    data_with_Background = pd.concat([Background,Normal,Botnet])
    data_with_Background['packet'] = packet_num
    data_without_Background = pd.concat([Normal,Botnet])
    data_without_Background['packet'] = packet_num
    
    return data_with_Background,data_without_Background,sub_data

def count_elements(nested_list):
    count = 0
    for element in nested_list:
        count += len(element)
    return count

def ReadCTU13():
    #order_sequence = [1, 2, 9, 5, 13, 6, 7, 8, 12, 3, 4, 10, 11]
    order_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    ordered_data = []
    counts = []
    for i in order_sequence:
        data_with_Background,data_without_Background,sub_data = load_data(i)
        shuffled_df = data_without_Background.sample(frac=1).reset_index(drop=True)
        ordered_data.append(shuffled_df)
        counts.append(count_elements(ordered_data))
        
        '''
        if i == 9: 
            print("section 1,2,9 at:",count_elements(ordered_data))
        elif i == 13:
            print("section 5,13 at:",count_elements(ordered_data))
        elif i == 6:
            print("section 6 at:",count_elements(ordered_data))
        elif i == 7:
            print("section 7 at:",count_elements(ordered_data))
        elif i == 8:
            print("section 8 at: ",count_elements(ordered_data))
        elif i == 11:
            print("section 3,4,10,11 at: ",count_elements(ordered_data))
        elif i == 12:
            print("section 12 at:",count_elements(ordered_data))
        '''    

    result_df = pd.concat(ordered_data, ignore_index=True)
    X_train = result_df.drop("Label",axis=1).values
    y_train = result_df["Label"].values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    return X_train,y_train,counts


def uniform_sampling(X,y,label_budget):
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=label_budget, random_state=42)
    return X_sample,y_sample

def uncertainty_samlping(X,y,best_program,label_budget):
    y_proba = best_program.predict_proba(X)
    uncertainty_scores = y_proba.max(axis=1)
    sorted_indices = uncertainty_scores.argsort()
    num_samples_to_label = int(len(X)*label_budget)
    labeled_indices = sorted_indices[-num_samples_to_label:]
    X_sample = X[labeled_indices]
    y_sample = y[labeled_indices]
    return X_sample,y_sample

def archive_policy(X,y,window_size):
    X_subset = []
    y_subset = []
    min_samples_per_label = int(window_size/2)
    for label in range(2):
        label_indices = [i for i, label_value in enumerate(y) if label_value == label]
        np.random.shuffle(label_indices)
        sampled_indices = label_indices[:int(min_samples_per_label)]
        
        if len(label_indices) < min_samples_per_label:
            for _ in range(int(min_samples_per_label - len(label_indices))): 
                random_index = np.random.choice(label_indices)  # Corrected variable name
                sampled_indices.append(random_index)
                
        for i in sampled_indices:
            X_subset.append(X[i])
            y_subset.append(y[i])
    return X_subset,y_subset

def plot_results(accuracy,kappa,kappa_plus,kappa_pp,macro_f1,weighted_f1,count_index,sample_method,label_budget,counts):
    model_string = "TPG"

    # Create a dictionary to hold the data
    data = {
        'accuracy': accuracy,
        'kappa': kappa,
        'kappa_plus': kappa_plus,
        'kappa_pp':kappa_pp,
        'macro_f1': macro_f1,
        'weighted_f1':weighted_f1
    }
    df = pd.DataFrame(data, index=count_index)
    df.to_csv(model_string+" "+sample_method+ " " +str(label_budget)+'.csv')
    #plt.figure()

    plt.figure(figsize=(30,10))
    plt.subplot(2, 1, 1)
    # Plotting the line graph
    plt.plot(count_index, accuracy, label='accuracy')
    plt.plot(count_index, kappa, label='kappa')

    plt.plot(count_index, kappa_pp, label='kappa++')
    
    '''
    plt.axvline(x=316363, color='red', linestyle='--', linewidth=2)
    plt.text(10, 0, '1,2,9', bbox=dict(facecolor='white', alpha=0.5))
    plt.axvline(x=393885, color='red', linestyle='--', linewidth=2)
    plt.text(10, 0, '5,13', bbox=dict(facecolor='white', alpha=0.5))
    plt.axvline(x=406009, color='red', linestyle='--', linewidth=2)
    plt.text(10, 0, '6', bbox=dict(facecolor='white', alpha=0.5))
    plt.axvline(x=407749, color='red', linestyle='--', linewidth=2)
    plt.text(10, 0, '7', bbox=dict(facecolor='white', alpha=0.5))
    plt.axvline(x=486698, color='red', linestyle='--', linewidth=2)
    plt.text(10, 0, '8', bbox=dict(facecolor='white', alpha=0.5))
    plt.axvline(x=496494, color='red', linestyle='--', linewidth=2)
    plt.text(10, 0, '12', bbox=dict(facecolor='white', alpha=0.5))
    plt.axhline(0, color='red', linewidth=1) 
    '''
    for i in range(len(counts)):
        plt.axvline(x=counts[i], color='red', linestyle='--', linewidth=2)
        plt.text(10, 0, str(i), bbox=dict(facecolor='white', alpha=0.5))
        
    plt.grid()
    plt.ylabel('Value')
    #plt.ylim(-1.5, 1.5)
    plt.title(model_string)
    plt.legend()

    plt.subplot(2, 1, 2) 
    plt.plot(count_index, accuracy, label='accuracy')
    plt.plot(count_index, macro_f1, label='macro')
    plt.plot(count_index, weighted_f1,label ='weighted')
    plt.grid()
    for i in range(len(counts)):
        plt.axvline(x=counts[i], color='red', linestyle='--', linewidth=2)
        plt.text(10, 0, str(i), bbox=dict(facecolor='white', alpha=0.5))
        
    '''
    plt.axvline(x=316363, color='red', linestyle='--', linewidth=2)
    plt.text(10, 0, '1,2,9', bbox=dict(facecolor='white', alpha=0.5))
    plt.axvline(x=393885, color='red', linestyle='--', linewidth=2)
    plt.text(10, 0, '5,13', bbox=dict(facecolor='white', alpha=0.5))
    plt.axvline(x=406009, color='red', linestyle='--', linewidth=2)
    plt.text(10, 0, '6', bbox=dict(facecolor='white', alpha=0.5))
    plt.axvline(x=407749, color='red', linestyle='--', linewidth=2)
    plt.text(10, 0, '7', bbox=dict(facecolor='white', alpha=0.5))
    plt.axvline(x=486698, color='red', linestyle='--', linewidth=2)
    plt.text(10, 0, '8', bbox=dict(facecolor='white', alpha=0.5))
    plt.axvline(x=496494, color='red', linestyle='--', linewidth=2)
    plt.text(10, 0, '12', bbox=dict(facecolor='white', alpha=0.5))
    plt.axhline(0, color='red', linewidth=1) 
    '''
    
    # Adding labels and title to the plot
    plt.xlabel('Series')
    plt.ylabel('Value')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.savefig(model_string+" "+sample_method+ " " +str(label_budget)+'.png')

def champion_run(model,champion,X_train,y_train,sample_method,label_budget,max_samples,packet_num,counts):
    stream = DataStream(X_train,y=y_train)
    n_samples = 0
    correct_cnt = 0
    accuracy = []
    kappa = []
    kappa_plus = []
    kappa_pp = []
    count_index = []
    window_size = 200
    macro_f1 = []
    weighted_f1 = []
    i = 0
    while n_samples < max_samples and stream.has_more_samples():
        n_samples+=window_size
        X, y = stream.next_sample(window_size)
        predictions = model.predict(X,champion)
        score = accuracy_score(y, predictions)

        p0 = score
        pe = dummy_classifier(X,y)
        accuracy.append(p0)
        
        kappa.append(kappa_score(y,predictions))
        kappa_plus.append(kappa_plus_score(p0,pe))
        kappa_pp.append(kappa_PP_score(p0,pe))
        macro,weighted = Fmeasure(y,predictions)
        macro_f1.append(macro)
        weighted_f1.append(weighted)
        count_index.append(n_samples)

    plot_results(accuracy,kappa,kappa_plus,kappa_pp,macro_f1,weighted_f1,count_index,"champion"+packet_num+sample_method,label_budget,counts)

def run_model_with_tpg(X_train, y_train, label_budget, sample_method,max_samples,counts):    
    Parameters.ACTIONS = [0,1]
    Parameters.NUM_OBSERVATIONS = 9
    Parameters.POPULATION_SIZE = 13
    Parameters.INITIAL_PROGRAM_POPULATION =50
    #Parameters.LUCKY_BREAK_NUM = 2
    model = Model()

    debugger = Debugger()
    # Rest of your setup code (like setting window size, max samples, etc.) goes here...
    stream = DataStream(X_train, y=y_train)
    n_samples = 0
    window_size = 200
    
    #Add in Subset sampling policy and archiving policy
    subset_X = X_train[:window_size]
    subset_y = y_train[:window_size]
    subset_y = subset_y.tolist()
    subset_X = subset_X.tolist()
    gen = 0
    accuracy = []
    kappa = []
    kappa_plus = []
    kappa_pp = []
    macro_f1 =[]
    weighted_f1 = []
    count_index = []
    champion_index = []
    for i in range(len(counts)):
        c_in = int(counts[i]/window_size)*window_size
        print(i,": ",c_in)
        champion_index.append(c_in)

    while n_samples < max_samples and stream.has_more_samples():
        n_samples+=window_size
        print("n_samples processed: ", n_samples)
        X, y = stream.next_sample(window_size)
        # TPG agent selection and evaluation

        if gen != 0:
            predictions = model.predict(X,champion_team)
            score = accuracy_score(y,predictions)
            print("Champion's prediction is:",score)
            
            p0 = score
            pe = dummy_classifier(X,y)
            accuracy.append(p0)
            kappa.append(kappa_score(y,predictions))
            kappa_plus.append(kappa_plus_score(p0,pe))
            kappa_pp.append(kappa_PP_score(p0,pe))
            macro,weighted = Fmeasure(y,predictions)
            macro_f1.append(macro)
            weighted_f1.append(weighted)
            count_index.append(n_samples)

      
        if sample_method == "random":         
            X_sample,y_sample = uniform_sampling(X,y,label_budget)

        for item in X_sample.tolist():
            subset_X.append(item)

        for item in y_sample.tolist():
            subset_y.append(item)

        subset_X,subset_y = archive_policy(subset_X,subset_y,window_size)
        
        for i in range(10):
            champion_team=model.generation(X,y,gen)
        
        if n_samples in champion_index:
            teampath="champion_team_"+str(n_samples)
            modelpath = "champion_model_"+str(n_samples)
            model.saveChampionModel(modelpath)
            model.saveChampionTeam(champion_team,teampath)
        
        gen = gen+10    
        print("Generation: ",gen)
        print("Team Numbers: ",len(model.teamPopulation))
        #model.print_team_hierarchy()

        

    plot_results(accuracy,kappa,kappa_plus,kappa_pp,macro_f1,weighted_f1,count_index,sample_method,label_budget,counts)
    for i in range(len(champion_index)):
        model_path = "champion_model_"+str(champion_index[i])
        team_path = "champion_team_"+str(champion_index[i])
        champion_model = model.loadChampionModel(model_path)
        champion_team = model.loadChampionTeam(team_path)
        champion_run(champion_model,champion_team,X_train,y_train,sample_method,label_budget,max_samples,str(i))        


def main():
    np.seterr(all="ignore")
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_budget',default=0.01)
    parser.add_argument('--sample_method',default="random")

    args = parser.parse_args()
    label_budget = args.label_budget
    sample_method = args.sample_method
    X_train,y_train,counts = ReadCTU13()
    #X_train,y_train = ReadISOT()
    #max_samples = 2000
    max_samples = 801000
    #X_train,y_train = ReadElec()
    run_model_with_tpg(X_train, y_train, float(label_budget), sample_method,max_samples,counts)
    
    '''
    champion_index = []
    window_size = 2000
    Parameters.ACTIONS = [0,1]
    Parameters.NUM_OBSERVATIONS = 9
    Parameters.POPULATION_SIZE = 30
    Parameters.INITIAL_PROGRAM_POPULATION =100
    Parameters.LUCKY_BREAK_NUM = 1
    model =Model()
    for i in range(len(counts)):
        c_in = int(counts[i]/window_size)*window_size
        print(i,": ",c_in)
        champion_index.append(c_in)
    for i in range(len(champion_index)):
        model_path = "champion_model_"+str(champion_index[i])
        team_path = "champion_team_"+str(champion_index[i])
        champion_model = model.loadChampionModel(model_path)
        champion_team = model.loadChampionTeam(team_path)
        champion_run(champion_model,champion_team,X_train,y_train,sample_method,label_budget,max_samples,str(i),counts)     

'''

if __name__ == '__main__':
    main()