#imports
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from collections import Counter
import string
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.cluster import dbscan
from sklearn.neighbors import NearestNeighbors
from factor_analyzer import FactorAnalyzer
from Levenshtein import jaro_winkler
from kneed import KneeLocator
import nltk
from nltk.collocations import *
from nltk.tokenize import word_tokenize


#return the data for a specified task and event
def get_task_df(task_name, event_df):
    return event_df.loc[event_df['task']==task_name]

# reads the 300_wf.csv file and transforms it into action traces usable by the TEP
# chosen_event can be 'ogame' #'ecsc' #'hitb' #'ekoparty'
def read_ascend_ctf_data(cooked_directory, chosen_event):
    path_to_wf_csv = cooked_directory+"300_wf.csv"#path to read the merged data as a csv
    task_data_df = read_parsed_csv(path_to_wf_csv)

    #split data into events
    hitb_data = task_data_df[task_data_df['event']=='hitb']
    ecsc_data = task_data_df[task_data_df['event']=='ecsc']
    ekoparty_data = task_data_df[task_data_df['event']=='ekoparty']
    ogames_data = task_data_df[task_data_df['event'].str.contains('ogame')] #pools together all ogames

    if chosen_event == 'hitb':
        task_data_df = hitb_data
    elif chosen_event == 'ekoparty':
        task_data_df = ekoparty_data
    elif chosen_event == 'ecsc':
        task_data_df = ecsc_data
    elif chosen_event == 'ogame': #pools together all ogames
        task_data_df = ogames_data

    return task_data_df

#reads the 200_wf.csv file (merged data) and parses it into a dataframe suitable for use in experiments
def read_parsed_csv(path_of_saved_csv):
    df = pd.read_csv(path_of_saved_csv)
    task_data_df = df
    
    #read all values and turn to string
    task_data_df['command'] = task_data_df['command'].astype(str)
    task_data_df['action'] = task_data_df['command type'].astype(str)
    task_data_df['task'] = task_data_df['challenge'].astype(str)
    task_data_df['clipped'] = task_data_df['clipped'].astype(str) 
    task_data_df['process'] = task_data_df['process'].astype(str)
    task_data_df['path for login'] = task_data_df['path for login'].astype(str)
    task_data_df['request substr'] = task_data_df['request substr'].astype(str)
    task_data_df = task_data_df.rename(columns={'timestamp': 'time'}) #the name had to fit other mulder file names, but didn't want to rename it in my notebooks
    task_data_df['time'] = pd.to_datetime(task_data_df['time'], format="ISO8601", utc=True) #format time into DT objects
    
    #originally had to manually align time formats before they fixed it
    '''task_data_df['time'] = task_data_df['time'].astype(str)
    time_series = task_data_df['time']
    times = []
    for time in time_series:
        time_str = time.split('.')[0].split('+')[0]
        if 'T' in time_str:
            datetime_object = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
        else:
            datetime_object = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        times.append(datetime_object)
    task_data_df['time'] = times
    print(f'new datatype\n\n\n{task_data_df['time']}')
    '''
    
    return task_data_df

#returns the 100_bias file with the specified bias columns (plus the P#)
def get_bias_df(biases):
    bias_df = pd.read_csv('../data/cooked/100_bias_data.csv')
    biases.append('participant')
    bias_df = bias_df[biases]
    biases.remove('participant')

    return bias_df


#returns all bad actions (that should not be considered in experiments), as well as the actions that ARE allowed in experiments
#dir_to_clean_actions should lead to a csv with a csv with a column named 'cls_command_normalized' that holds all allowed action types
def get_bad_actions(all_action_types_series, dir_to_clean_actions=None):
    
    all_action_types = all_action_types_series.unique() #get all unique action types
    
    #NOTE: these bad actions are not really used now, other than the stopwords. It is now set up to manually only include actions that are defined in 200_terminal_features.
    bad_actions = ['nan','', 'MagicW@nd456', '^C', #<- this has a sample of the bad commands that i manually found in CTF data (deleted the rest because of PII)
       'firefox_decrypt-1.1.1/firefox_decrypt.py', '"', #a bad command, defined by me, is a command that i cannot find by searching "{command} command line command" in Google
       'crunch5','dirbb','dirbuster-h', 'pid(B', #or that is obviously a mistake (like "Hydra" or "hyda" instead of "hydra")
       'debug(B', 'hot*ConEmu*', 'http://', 'hy', 'hydrl', 'GET""', 'base', '64',
       
       #stop words
       #fake "bad" terms that are real, but they don't actually give much context to the data analysis
       'pip', #we don't care if they installed a package
       'sudo', #not actually a command, use the command after
       'su', # not actually a command 
       'clear', #just clears screen, doesn't solve anything
       'cd', #not an actual function to solve
       'ls', #not an actual function to solve
       'll',
       'l',
       'dir',
       'cat', #not an actual function to solve
       'sh', #just invokes a shell, not actually useful
       'vi', #not an actual function to solve
       'vim', #not an actual function to solve
       'nvim',
       'subl',
       'nano', #not an actual function to solve
       'mousepad', #not an actual function to solve
       'less', #not an actual function to solve
       'echo', #just echoes what the user types, not helpful
       ]
    
    if dir_to_clean_actions != None:
        terminal_features_df=pd.read_csv(dir_to_clean_actions) #get the terminal features
        command_features = terminal_features_df['cls_command_normalized'].unique() #get all allowed/good commands from features
        allowed_action_types = [action_type for action_type in all_action_types if action_type not in bad_actions and action_type in command_features] #only allow actions not in bad_actions and that are in command_features
    else:
        allowed_action_types = all_action_types

    #this ensures that anything that isn't selected as good, is also delivered as "bad". Mostly pertinent to subtasks, as it takes in bad actions that should be skipped.
    for action_type in all_action_types:
        if action_type not in allowed_action_types and action_type not in bad_actions:
            bad_actions.append(action_type)

    return bad_actions, allowed_action_types

# cleaning dict, where whatever is on the left gets transformed into what's on the right
# used e.g. for python3 to also just count as python and not two separate actions
def get_cleaning_dict():
    cleaning_dict = {
        'python3': 'python',
        r'^[^_]*.sh': '.sh',
    }
    
    return cleaning_dict
   
def clean_terms(data_df):
    data_df['action'] = data_df['action'].str.replace(r'^[^_]*.sh', '.sh', regex=True)
    data_df['action'] = data_df['action'].str.replace('python3', 'python', regex=True)
    
    return data_df
    

#=========================================================

#                CLUSTERING (FA & EDIT DISTANCE)

#=========================================================

def get_parameters(event):
    if event == 'ecsc':
        ecsc_params = {
            'TF_threshold': #term frequencies must be higher than this number to be included in PAF clustering, else are left out
                {
                    'cult-1': 5,
                    'cult-5': 3,
                    'loss-1b': 4,
                    'loss-1a': 5,
                    'rep-8a': 4,
                    'rep-8b': 3,
                },
            'terms_to_concat_dict': #some terms are perfectly correlated (used the same amount of times in the same runs), thus are concatted in runs to allow PAF to occur)
                {
                    'cult-1': {},
                    'cult-5': {},
                    'loss-1b': {},
                    'loss-1a': {},
                    'rep-8a': {},
                    'rep-8b': {},
                },
            'n': #runs have to be longer than n to be considered in data
                {
                    'cult-1': 3,
                    'cult-5': 2,
                    'loss-1b': 3,
                    'loss-1a': 3,
                    'rep-8a': 2,
                    'rep-8b': 3,
                },
        }
        
        return ecsc_params
    elif event == 'ekoparty':
        ekoparty_params = {
            'TF_threshold': #term frequencies must be higher than this number to be included in PAF clustering, else are left out
                {
                    'cult-1': 2,  
                    'cult-5': 2, #any higher and there would only be 3 commands, any lower and there are more commands than participants (resulting in singular matrix)
                    'conf-5b': 5, 
                    'conf-5a': 2, 
                    'anch-1b': 2,
                    'anch-1a': 3,
                },
            'terms_to_concat_dict': #some terms are perfectly correlated (used the same amount of times in the same runs), thus are concatted in runs to allow PAF to occur)
                {
                    'cult-1': 
                        {
                            """'tcpdump': 'tcpdumppingnslookupcrunchifconfig',
                            'ping': 'tcpdumppingnslookupcrunchifconfig',
                            'nslookup': 'tcpdumppingnslookupcrunchifconfig',
                            'crunch': 'tcpdumppingnslookupcrunchifconfig',
                            'ifconfig': 'tcpdumppingnslookupcrunchifconfig',
                            'gzip': 'gzipffuf',
                            'ffuf': 'gzipffuf',""" #auto-prunes now
                        },
                    'cult-5': 
                        {
                            """'wfuzz': 'wfuzzgunzip',
                            'gunzip': 'wfuzzgunzip',"""
                        },
                    'conf-5b': 
                        {
                            """'wfuzz': 'wfuzzgobuster',
                            'gobuster': 'wfuzzgobuster',"""
                        },
                    'conf-5a': {
                        """'dig': 'digdate',
                        'date': 'digdate',
                        'ifconfig': 'chmodifconfignetstat',
                        'netstat': 'chmodifconfignetstat',
                        'gcc': 'gccmake',
                        'make': 'gccmake',
                        'chmod': 'chmodifconfignetstat',
                        'groups': 'groupsss',
                        'ss': 'groupsss',"""
                    },
                    'anch-1b': {},
                    'anch-1a': 
                        {
                            """'python': 'pythonsearchsploit',
                            'searchsploit': 'pythonsearchsploit',""" 
                        },
                },
            'n': #runs have to be longer than n to be considered in data
                {
                    'cult-1': 2, # 3 makes it go from 27 down to 10 participants
                    'cult-5': 1, #most participants have zero commands, let alone multiple
                    'conf-5b': 3,
                    'conf-5a': 2, # too few runs at 3, becomes singular
                    'anch-1b': 1,
                    'anch-1a': 2, #gives almost 10 more participants
                },
        }
    elif event == 'ogame':
        ekoparty_params = {
            'TF_threshold': #term frequencies must be higher than this number to be included in PAF clustering, else are left out
                {
                    'rep-8b': 10, #tried 8, just muddles some clusters at weird split points
                    'rep-8a': 8,
                    'loss-1b': 14, #lower muddles the clusters and makes weird splits
                    'loss-1a': 7,
                    'cult-6b': 14, #lots of aimless use of some commands, leading to many uses of some commands that are not pertinent
                    'cult-6a': 9,
                    'conf-5b': 7, 
                    'conf-5a': 18, #lower thresholds were letting in too many terms that were muddling the clusters, there could be an argument to remove even more, but that might delete too many actions from some runs
                    'anch-1b': 7,
                    'anch-1a': 10,
                },
            'terms_to_concat_dict': #some terms are perfectly correlated (used the same amount of times in the same runs), thus are concatted in runs to allow PAF to occur)
                {
                    'rep-8b': {},
                    'rep-8a': {},
                    'loss-1a': {},
                    'cult-6b': {},
                    'cult-6a': {},
                    'conf-5b': {},
                    'conf-5a': {},
                    'anch-1b': {},
                    'anch-1a': {},
                },
            'n': #runs have to be longer than n to be considered in data
                {
                    'rep-8b': 3, #1: 77 parts, 2: 69, 3: 60, 4: 46, 5: 42 parts
                    'rep-8a': 3, #1: 110 parts, 2: 99, 3: 92, 4: 79, 5: 68 parts
                    'loss-1b': 5, #1: 105 parts, 5: 85 parts, 10: 71
                    'loss-1a': 2, #1: 130 parts, 2: 119, 3: 76, 5: 52 parts
                    'cult-6b': 3, #1: 110 parts, 2: 95, 3: 78, 4: 66, 5: 60 parts
                    'cult-6a': 2, #1: 46 parts, 2: 38, 3: 29, 5: 19 parts, 10: 9 parts
                    'conf-5b': 10, #1: 121 parts, 5: 108 parts, 10: 96 parts
                    'conf-5a': 5, #1: 138 parts, 2: 124, 3: 108, 5: 99 parts, 10: 73 parts
                    'anch-1b': 5, #1: 173 parts, 5: 161 parts, 10: 125 parts
                    'anch-1a': 5, #1: 139 parts, 5: 130 parts, 6: 121 parts, 7:116 parts
                },
    }
            
        return ekoparty_params
    
    else: #custom set for testing, if you want to make it permenant, then add a new entry using above events as an example
        #TODO change "custom_task" to match whatever the task name is in the csv file
        custom_params = {
            'TF_threshold': #term frequencies must be higher than this number to be included in PAF clustering, else are left out
                {
                    'custom_task': 0,
                },
            'terms_to_concat_dict': #some terms are perfectly correlated (used the same amount of times in the same runs), thus are concatted in runs to allow PAF to occur)
                {
                    'custom_task': {},
                },
            'n': #runs have to be longer than n to be considered in data
                {
                    'custom_task': 10, 
                },
    }
            
        return custom_params



    

#returns counter with all frequency of each term in runs in the given task 
def get_TF_from_runs_in_task(task, actions_to_include, task_data_df, participants_to_exclude = []):
    
    term_counter = Counter() #holds counts
    task_data_df = get_task_df(task, task_data_df) #get the data for this task
    for participant in task_data_df['participant'].unique(): #go through every participant in task
        if participant not in participants_to_exclude: #if the participant is included


            participant_data_df = task_data_df.loc[task_data_df['participant'] == participant] #get participant data
            
            new_row = [] #initialize new row
            for action_type in actions_to_include: #go trhough every command type included
                num_of_this_cmd = participant_data_df.loc[(participant_data_df['action'] == action_type) & (participant_data_df['action'] != 'nan')].shape[0] #get count of this action for this run
                term_counter[action_type] += num_of_this_cmd
    
    return term_counter

#gathers all runs that have more than n actions in them (to remove runs with little going on)
def get_runs_with_more_than_n_actions(actions_to_include, task, n, terms_to_concat_dict, task_data_df, participants_to_exclude = []):
    runs = [] #will hold the gathered runs
    participants_included = [] #these will be the participants that had runs longer than n
    task_data_df = get_task_df(task, task_data_df) 
    for participant in task_data_df['participant'].unique(): #go through every participant in task
        if participant not in participants_to_exclude: #if the participant is included

            participant_data_df = task_data_df.loc[task_data_df['participant'] == participant] #get participant data
            
            
            participant_actions = participant_data_df['action']
                
            run = [action for action in participant_actions if action in actions_to_include] #take out bad actions
            #concatenate terms that are to be merged (from concat_dict). this is because these terms show up with the same exact frequency, making factor analysis impossible
            concat_correlated_terms_run = []
            for action in run:
                if action in terms_to_concat_dict.keys():
                    concat_correlated_terms_run.append(terms_to_concat_dict[action])
                else:
                    concat_correlated_terms_run.append(action)
            #check if the run is longer than n and save it if it is
            if len(concat_correlated_terms_run) >= n:
                runs.append(concat_correlated_terms_run)
                participants_included.append(participant)
    
    return runs, participants_included

#performs the intracluster analysis for creating a plot that can be used to find the knee for best number of clusters
def wss_calculation(K, data):
    WSS = []
    for i in range(K):
        cluster = AgglomerativeClustering(n_clusters= i+1)
        cluster.fit_predict(data)
        # cluster index
        label = cluster.labels_
        wss = []
        for j in range(i+1):
            # extract each cluster according to its index
            idx = [t for t, e in enumerate(label) if e == j]
            cluster = data[idx,]
            # calculate the WSS:
            cluster_mean = cluster.mean(axis=0)
            distance = np.sum(np.abs(cluster - cluster_mean)**2,axis=-1)
            wss.append(sum(distance))
        WSS.append(sum(wss))
    return WSS

#stablizes and standardizes the term frequency counts of actions in runs
def get_stabilized_standardized_TF_from_runs(runs):
    #get actions to go through
    terms_to_go_through = np.unique([action for run in runs for action in run]) #get all actions within these runs (different from all allowed actions)
    rows = []
    #get term frequency counts for each run
    for run in runs:
        new_row = []
        for term in terms_to_go_through:
            new_row.append(sum([1 if action == term else 0 for action in run]))
        rows.append(new_row)
    
    #turn TF data into a dataframe
    stabilized_standardized_TF_df = pd.DataFrame(rows, columns=terms_to_go_through)
    
    #stabilize using square root transformation (log of zero is undefined/-inf, so can't use log transformation)
    for term in terms_to_go_through:
        stabilized_standardized_TF_df[term] = np.sqrt(stabilized_standardized_TF_df[term])
    
    #standardize (zero-mean and unit variance)
    for term in terms_to_go_through:
        stabilized_standardized_TF_df[term] = (stabilized_standardized_TF_df[term]-stabilized_standardized_TF_df[term].mean())/stabilized_standardized_TF_df[term].std()
    
    return stabilized_standardized_TF_df

#plots the dendrogram of the hierarchical clusters found from factor analysis
def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

#performs the hierarchical clustering of runs based on their rotated factor scores
def factor_analysis_clustering(task_data_df, chosen_task, all_action_types, parameters, verbose = False, save=None):
    
    #the task to run on (new variable because i was originally too lazy to change all instances of "task" to "chosen_task")
    task = chosen_task
    
    #all commands in the task
    actions_in_task = task_data_df[task_data_df['task'] == task]['action'].unique()

    #these are the commands that will be included
    actions_to_include = [cmd for cmd in actions_in_task if cmd in all_action_types]

    #get the term frequency counts of each individual action
    term_counter = get_TF_from_runs_in_task(task, actions_to_include, task_data_df)
    
    #include only the actions with a TF greather than the threshold
    TF_threshold = parameters['TF_threshold'][chosen_task] 
    actions_to_include_in_task = []
    for key, value in term_counter.items():
        if value >= TF_threshold:
                actions_to_include_in_task.append(key)

    #NOTE: the below code can help to automate TF_threshold if it becomes an issue, i left it out as ideally one should check this explicitly when setting the value
    #todo if use this: rename to TF_lower_threshold
    #take top n terms that can be taken without causing singular matrix in FA (will not change it if)
    #while len(actions_to_include_in_task) > len(runs): #there are more actions than runs, will go until it finds a suitable threshold, or until it finds that there is no possible suitable threshold 
       # print(f'more terms than runs ({len(actions_to_include_in_task)},{len(runs)}), changing TF threshold from {TF_threshold} to {TF_threshold+1}')
        #actions_to_include_in_task = []
        #TF_threshold += 1 #add one, see if we can find a better threshold
        #for key, value in term_counter.items():
            #if value >= TF_threshold:
                #actions_to_include_in_task.append(key)
        
       # if len(actions_to_include_in_task) < 2: #there are 1 or 0 actions included, this is not suitable
            #raise Exception('no suitable TF threshold to satisfy factor analysis')
                
    
    runs, participants_included, paf_data_df, n_factors_greater_than_1 = get_n_comps_and_prune_correlations(actions_to_include_in_task, task, task_data_df, parameters)
    
    #rotate factors
    #oblique rotations: direct oblimin & promax
    n_comps = n_factors_greater_than_1
    paf = FactorAnalyzer(n_factors=n_comps, rotation='oblimin')
    factors = paf.fit(paf_data_df)
    
    
    #transform data into rotated factor scores
    rotated_factor_scores = paf.transform(paf_data_df)
    
    
    #use elbow method to choose the number of clusters

    WSS=wss_calculation(rotated_factor_scores.shape[0], rotated_factor_scores)

    cluster_range = range(1, rotated_factor_scores.shape[0]+1)

    # get knee using the kneedle algorithm
    kneedle = KneeLocator(cluster_range, WSS,  S=1.0, curve="convex", direction="decreasing")
    
    if verbose: #actually plot the scree plot if verbose
        kneedle.plot_knee(xlabel='Number of clusters (k)',ylabel='Total intra-cluster variation',title='Optimal number of clusters')

        print(kneedle.knee)
    
    n_clusters = kneedle.knee #this is the optimal number of clusters
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(rotated_factor_scores) #now we can cluster using the optimal number to get final clusters

    if verbose or save:
        
        plt.clf()
        plt.title(f'{task} Hierarchical Clustering Dendrogram')
        plot_dendrogram(clustering, labels=clustering.labels_)
        if save: #save plot dendrogram of cluster if save
            plt.savefig(f'{save}')
        if verbose: #show plot dendrogram of cluster if verbose
            plt.show()
        
        
        
        
    if verbose:
        pca = PCA(n_components = 2)
        pca.fit(rotated_factor_scores)
        reduced_dimenions = pca.transform(rotated_factor_scores)
        
        colors = cm.rainbow(np.linspace(0, 1, clustering.n_clusters_)) #get n_clusters colors to plot
        plt.scatter(reduced_dimenions[:, 0], reduced_dimenions[:, 1], alpha=0) #scatter sso we have a plot the size of points (was plotting text points outside plot otherwise)
        for i, point in enumerate(reduced_dimenions): #plot points as their cluster label
            plt.text(point[0], point[1], clustering.labels_[i], label = clustering.labels_[i], fontsize=10, color=colors[clustering.labels_[i]])
        plt.title(f'{task} Clusters of PCA-2-Dim-Reduced Rotated Factor Scores')
        plt.show()
        
    return participants_included, clustering.labels_, runs, paf_data_df


def get_n_comps_and_prune_correlations(actions_to_include, task, task_data_df, parameters):
    #the dict that tells us which action terms to concat (due to being perfectly correlated)
    if task in parameters['terms_to_concat_dict'].keys() and parameters['terms_to_concat_dict'][task] != None:
        terms_to_concat_dict = parameters['terms_to_concat_dict'][task]
    else:
        terms_to_concat_dict = dict()

    for coin in range(len(actions_to_include)**2): #can only run enough times as it would take to concat every term (which would mean every term is highly correlated)
        
        #try once, then see if we need to keep going and find terms to concatenate

        try: #see if it works, if it does then the return statement will exit the while loop
            #get runs and list of participants in task that actually did something (len(cmd type)>n)
            n = parameters['n'][task]
            runs, participants_included = get_runs_with_more_than_n_actions(actions_to_include, task, n, terms_to_concat_dict, task_data_df,)

            #get tf-idf scores for every action in the actions to include and for participants that actually have good runs
            paf_data_df = get_stabilized_standardized_TF_from_runs(runs)
            
            
            #find the number of  factors by using the kaiser criterion (# of eigenvalues > 1). an eigenvalue of 1 means that the factor represents the same information as one variable 
            n_comps = paf_data_df.shape[1]-1
            paf_not_rot = FactorAnalyzer(n_factors=n_comps)
            factors = paf_not_rot.fit(paf_data_df) #NOTE would fail here and go to except
            eigenvalues, common_factor_eigenvalues = paf_not_rot.get_eigenvalues()
            n_factors_greater_than_1 = len([eigenvalue for eigenvalue in eigenvalues if eigenvalue > 1])            


            return runs, participants_included, paf_data_df, n_factors_greater_than_1 #it worked, let's exit the loop and return
        except Exception as e: #if we made it here then the singular matrix error occured and we need to concat some terms

            #1 get the most highly correlated terms
            corr = paf_data_df.corr() #correlation matrix of all terms
            highly_correlated_terms = dict()
            for col in corr.columns:
                for row_i in range(len(corr.columns)):
                    if corr[col].iloc[row_i] > .5: #if the correlation between two terms is greater than .5 (highly correlated)
                        row_name = corr.columns[row_i]
                        if col != row_name and col not in row_name.split('|') and row_name not in col.split('|'):
                            sorted_names = np.sort([row_name, col]) #sort so we don't double count the reversed terms in the matrix
                            highly_correlated_terms[f'{str(sorted_names[0])}:{str(sorted_names[1])}'] = float(corr[col].iloc[row_i]) #add to highly correlated terms dict

            highly_correlated_terms_dict = dict(sorted(highly_correlated_terms.items(), key=lambda item: item[1])) #sort the dict
            
            #2 check if any of the highly correlated terms are values in the dict
            #if they are, then concat the correlated term to the new correlated terms
            #i don't think we need to worry about two already concated terms being compared, as this consecutively builds (not building in parallel)
            
            #get the two highest correlated terms
            highly_correlated_terms = list(highly_correlated_terms_dict.keys())
            #print(highly_correlated_terms)
            highest_correlated_terms = highly_correlated_terms[-1].split(':') #last key is the highest correlation
            #print(highest_correlated_terms)
            high_term_1 = highest_correlated_terms[0]
            high_term_2 = highest_correlated_terms[1]

            term_in_concat_dict = 0 #0 if neither term is in the dict
            #check if either of the terms are an already concatenated term
            if high_term_1 in terms_to_concat_dict.values() and high_term_2 in terms_to_concat_dict.values():
                term_in_concat_dict = 3 #both
            elif high_term_1 in terms_to_concat_dict.values(): #look through values, which hold the concatenated terms
                term_in_concat_dict = 1 #first one only
            elif high_term_2 in terms_to_concat_dict.values():
                term_in_concat_dict = 2 #second one only
            

            #3 concat the two terms if they aren't values, else add to the concated value and the new key if already in concated term

            if term_in_concat_dict == 0: #neither term is a concatenated term
                concat_str = high_term_1+'|'+high_term_2
                terms_to_concat_dict[high_term_1] = concat_str
                terms_to_concat_dict[high_term_2] = concat_str
                keys = [] #this just makes it so we can skip the keys for-loop at the bottom
            elif term_in_concat_dict == 1: #1st term is a concatenated term
                #find the keys that have the concatenated term as a value
                keys = [key for key, value in terms_to_concat_dict.items() if value == high_term_1] #all the keys that have vallues equal to high_term_1
                keys.append(high_term_2) #this is technically a set, we get all terms in the concatenated value and then also add the term it was correlated with
            elif term_in_concat_dict == 2: #2nd term is a concatenated term
                #find the keys that have the concatenated term as a value
                keys = [key for key, value in terms_to_concat_dict.items() if value == high_term_2] #all the keys that have vallues equal to high_term_2
                keys.append(high_term_1) #this is technically a set, we get all terms in the concatenated value and then also add the term it was correlated with
            elif term_in_concat_dict == 3: #both are concatenated term
                keys_1 = set([key for key, value in terms_to_concat_dict.items() if value == high_term_1]) #all the keys that have vallues equal to high_term_1
                keys_2 = set([key for key, value in terms_to_concat_dict.items() if value == high_term_2]) #all the keys that have vallues equal to high_term_2
                keys = keys_1.union(keys_2) #get set of all keys
            
            #update keys with concatenated str
            concat_str = '|'.join(keys)
            for key in keys:
                terms_to_concat_dict[key] = concat_str

        
    #if we got this far, we couldn't fix it, so let them know and hault program
    raise Exception('Cannot find a terms_to_concat_dict that satisfies factor analysis')



            


def edit_distance_clustering(participants_included, labels, runs, paf_data_df, verbose=False):
    
    #getting the runs, now with cluster labels from PAF clustering
    rows = []
    for i in range(len(participants_included)): #make new rows with participant, cluster label, and run
        participant = participants_included[i]
        label = labels[i]
        run = runs[i]
        rows.append([participant, label, run])
    runs_with_paf_labels = pd.DataFrame(rows, columns=['participant','paf_cluster_label','run'])
    
    
    #get dictionary that map actions to symbols
    symbols = string.printable #get all ASCII printable characters in python
    terms_included_in_task = list(paf_data_df.columns) #all termns that were used to cluster in this task
    term_to_symbol = dict()

    for i,term in enumerate(terms_included_in_task): #make a dictionary of string action: symbol representing string action
        term_to_symbol[term] = symbols[i]
        
    #turn arrays of string actions into strings of symbol actions
    runs_as_strings = []
    for run in runs_with_paf_labels['run']: #for each run we used
        run_as_string = ''
        for term in run: #turn the run into a string using symbols
            run_as_string += term_to_symbol[term]
        runs_as_strings.append(run_as_string)
    runs_with_paf_labels['string_run'] = runs_as_strings #add new column
    
    results = []

    for label in runs_with_paf_labels['paf_cluster_label'].unique(): #for each cluster found in PAF clustering
        rows_with_label = runs_with_paf_labels[runs_with_paf_labels['paf_cluster_label'] == label] #get the runs that match this cluster label
        #print(rows_with_label)
        runs_with_label = list(rows_with_label['string_run']) #get the string runs
        
        #define the distance metric for dbscan
        #uses 1-jaro-winkler as distance metric. jaro-winkler gives similarity score normalized between 0 and 1.
        #intuitively, two strings that are compositionally similar will have a lower (closer) distance
        def jaro_winkler_metric(x, y): 
            i, j = int(x[0]), int(y[0])     # extract indices
            return 1-jaro_winkler(runs_with_label[i], runs_with_label[j]) #get distance
        
        X = np.arange(len(runs_with_label)).reshape(-1, 1) #get indices of runs to feed into dbscan
        min_pts = 2
        #min_pts = 2*len(terms_included_in_task) #good rule of thumb, but not enough samples to use this method
        
        
        #find nearest neighbors
        epsilon = 0.15 #TODO find this default some other way? actually doesn't matter, since epsilon gets changed anyways or isn't used if less than min_pts
        if len(X) > min_pts:
            k = min_pts
            neighbors = NearestNeighbors(n_neighbors=k, n_jobs=1, metric=jaro_winkler_metric)
            neighbors.fit(X)
            dists, inds = neighbors.kneighbors(X)
            distances = [dist[k-1] for dist in dists]
            distances = np.sort(distances)[::-1]
            #print(distances)
            cluster_range = range(1, len(distances)+1)
            kneedle = KneeLocator(cluster_range, distances,  S=1.0, curve="concave", direction="decreasing")
            if verbose:
                kneedle.plot_knee(title=f'knee {label} : {kneedle.knee_y}')
            knee_epsilon = kneedle.knee_y
            '''print(f'knee {label}: {epsilon}')
            print(distances)'''
            
            kneedle = KneeLocator(cluster_range, distances,  S=1.0, curve="convex", direction="decreasing")
            if verbose:
                kneedle.plot_knee(title=f'elbow {label} : {kneedle.knee_y}')
            elbow_epsilon = kneedle.knee_y
            '''print(f'elbow {label}: {epsilon}')
            print(distances)'''
            if knee_epsilon and elbow_epsilon:
                epsilon = max(knee_epsilon, elbow_epsilon)
            elif knee_epsilon:
                epsilon = knee_epsilon
            elif elbow_epsilon:
                epsilon = elbow_epsilon
            else:
                epsilon = np.median(distances)
            
            if epsilon == distances[-1]:
                epsilon = np.median(distances)
                if epsilon == 0.0:
                    epsilon = .2 #median is 0, so just set a random epsilon to get DBSCAN to just start
        elif len(X) == min_pts:
            epsilon = .2 #means that their similarity is .8 #TODO empirical way to compute/decide this?
            
        core_samples, db_labels = dbscan(X, metric=jaro_winkler_metric, eps=epsilon, min_samples=min_pts) #run dbscan using custom distance metric
        
        #print out results for each PAF cluster label
        if verbose:
            print(f'-------------------------------------------\nPAF_LABEL: {label} EPSILON: {epsilon}\n-----------------------------------')
        for db_label in np.unique(db_labels):
            if verbose:
                print(f'______DB_LABEL: {db_label}_____')
            for i, all_db_label in enumerate(db_labels):
                if all_db_label == db_label:
                    if verbose:
                        print(f"{list(rows_with_label['run'])[i]}")
                    new_result = [list(rows_with_label['participant'])[i], list(rows_with_label['paf_cluster_label'])[i], db_label]
                    results.append(new_result)
                    
    #results are participant, FA cluster, dbscan cluster                
    return results

#runs the full pipeline and returns the results
def get_pipeline_results(task_data_df, bad_actions, all_action_types, parameters, chosen_task, subtasks, verbose = None, save_json_directory=None):
    
    if save_json_directory: #saves this task's histrogram for use in the Rust GUI
        save_directory = f'{save_json_directory}/images/dendrogram_{chosen_task}.png'
        participants_included, labels, runs, paf_data_df = factor_analysis_clustering(task_data_df, chosen_task, all_action_types, parameters, verbose=verbose, save=save_directory)
    else: #otherwise just run the pipeline without saving
        participants_included, labels, runs, paf_data_df = factor_analysis_clustering(task_data_df, chosen_task, all_action_types, parameters, verbose=verbose)
    clustering_results = edit_distance_clustering(participants_included, labels, runs, paf_data_df)
    
    subtasks, tokenized_runs, participants, raw_runs = get_subtasks(task_data_df, chosen_task, bad_actions, subtasks)
    
    #encode the runs
    runs = encode_runs(chosen_task, subtasks, tokenized_runs, participants, raw_runs)
    
    #save results in an array for each participant's run
    #saved as [run_object, BOT_label, ECHO_label]
    pipeline_result = []
    for run in runs:
        for clustering_result in clustering_results:
            if run.participant == clustering_result[0]:
                FA_label = clustering_result[1]
                edit_dist_label = clustering_result[2]
                new_pipeline_result = [run, FA_label, edit_dist_label]
                pipeline_result.append(new_pipeline_result)

                break
            
    task_allowed_actions = paf_data_df.columns
    
    return pipeline_result, subtasks, task_allowed_actions

    
    
    

#=========================================================

#                SUBTASKS

#=========================================================

#return the data as a dataset of [run, task]
#a run is the sequnce actions a participant took to solve a task
def get_runs_and_labels(df, bad_actions):
    runs_and_labels = []
    for task in df['task'].unique(): #separate/save runs with their task label
        task_data_df = df[df['task'] == task] #get the data for the current task
        for participant in task_data_df['participant'].unique(): #go through every participant who has data in this task
            participant_data_df = task_data_df[task_data_df['participant'] == participant] #get the participant "run"
            action_type_history = [action_type for action_type in participant_data_df['action'] if action_type not in bad_actions] #take out bad actions/only include good actions
            if action_type_history and len(action_type_history) > 0: #if the history exists and has at least 1 action in its history (sometimes it comes back with null pointer or just a 0 length list)
                runs_and_labels.append([action_type_history,task, participant]) #append the run and its task label (and participant)
    return runs_and_labels

#returns the column of the collocation data as a 1-D array
def get_column(array, col_idx):
        return [row[col_idx] for row in array]

#finds if a collocation exists in the given tokenized run
def collocation_in_run(collocation, tokenized_run):
    try:
        index = 0
        first_token = collocation[0]
        while True:
            idx_of_first_token = tokenized_run.index(first_token, index)
            subsequence = tokenized_run[idx_of_first_token:idx_of_first_token+len(collocation)]
            if subsequence == collocation:
                return True
            index+=1
    except:
        return False

#defines all subtasks and returns the defined subtasks, the tokenized runs (sequentially repeated terms chopped), 
# the participants included, and the raw runs without chopped
def get_subtasks(chosen_set, chosen_task, bad_actions, subtasks = dict(), verbose = False):
    data = get_runs_and_labels(chosen_set, bad_actions)

    if verbose:
        #figure out how many runs we have in each task
        labels = [row[1] for row in data]
        unique,counts = np.unique(labels,return_counts=True)
        print(dict(zip(unique,counts)))
        
    #turn runs into string of words
    str_data = []
    for run, task, participant in data:
        run_str = ''
        for term in run:
            run_str += f'{term} '
        str_data.append([run_str, task, participant, run])
        
    # Construct long document of aggregated documents for chosen task
    document = ''
    participants = []
    raw_runs = []
    for str_run, task, participant, raw_run in str_data:
        if task == chosen_task:
            document += str_run
            participants.append(participant)
            raw_runs.append(raw_run)
    
    #get collocations
    
    word_tokens = word_tokenize(document)
    word_tokens = [token for token in word_tokens if token not in bad_actions] #remove bad terms
    
    #remove repeated tokens [hydra,hydra,hydra,...] -> [hydra]
    non_repeated_tokens = []
    current_token = ''
    for token in word_tokens:
        if token != current_token:
            non_repeated_tokens.append(token)
            current_token = token
            
    #get the collocations using nltk
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    tri_finder = TrigramCollocationFinder.from_words(non_repeated_tokens)

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    bi_finder = BigramCollocationFinder.from_words(non_repeated_tokens)

    quadgram_measures = nltk.collocations.QuadgramAssocMeasures()
    quad_finder = QuadgramCollocationFinder.from_words(non_repeated_tokens)
    
    
    #get postive PMI collocations
    quad_collocations = quad_finder.score_ngrams(quadgram_measures.pmi)
    pos_quad_collocations = [quad_collocation for quad_collocation in quad_collocations if quad_collocation[1]>0]
    tri_collocations = tri_finder.score_ngrams(trigram_measures.pmi)
    pos_tri_collocations = [tri_collocation for tri_collocation in tri_collocations if tri_collocation[1]>0]
    bi_collocations = bi_finder.score_ngrams(bigram_measures.pmi)
    pos_bi_collocations = [bi_collocation for bi_collocation in bi_collocations if bi_collocation[1]>0]
    
    if verbose:
        print(f'positive collocations for {len(pos_quad_collocations)} quadgrams')
        print(pos_quad_collocations)
        
        print(f'positive collocations for {len(pos_bi_collocations)} bigrams')
        print(pos_bi_collocations)
        
        print(f'positive collocations for {len(pos_tri_collocations)} trigrams')
        print(pos_tri_collocations)
        
    #combine collocations into one data structure
    
    pos_collocations = dict()
    pos_collocations[2] = get_column(pos_bi_collocations, 0)
    pos_collocations[3] = get_column(pos_tri_collocations, 0)
    pos_collocations[4] = get_column(pos_quad_collocations, 0)

    # get runs as tokens
    set_of_tokens_used = np.unique(non_repeated_tokens)
    tokenized_runs = []
    for run, task, participant in data:
        if chosen_task == task:
            tokenized_run = []
            current_token = ''
            for token in run:
                #if token == 'python3': #change python3 into python
                    #token = 'python'
                if token in set_of_tokens_used:
                    if token != current_token:
                        tokenized_run.append(token)
                        current_token = token
            tokenized_runs.append(tokenized_run)
    
    #get document frequency of collocations
    collocation_doc_freq = dict()
    for key in pos_collocations.keys():
        collocation_doc_freq[key] = Counter()
        collocations = pos_collocations[key]
        for tokenized_run in tokenized_runs:
            for collocation in collocations:
                if collocation_in_run(list(collocation), tokenized_run):
                    collocation_doc_freq[key][collocation] += 1
        collocation_doc_freq[key] = Counter({k: c for k, c in collocation_doc_freq[key].items() if c > 1}) #these are now subtasks (occur in more than one run)
        
    
    
    #define subtasks
    #define string representations of subtasks
    #subtasks = dict() #defined at function call or passed in
    
    for key in collocation_doc_freq.keys(): #for every length of ngram
        if key not in subtasks: #create dict if it was not passed in at function call
            subtasks[key] = dict()
        
        #find where we should start the naming process (starts at the maximum of the previously defined identifier plus one)
        coin = 0 #tracks which identifier we are on (ensures that we don't name new subtasks as previously named passed-in subtasks)
        if len(subtasks[key].values()) > 0:
            identifiers = []
            predefined_subtasks = subtasks[key].values()
            for predefined_subtask in predefined_subtasks:
                identifier = predefined_subtask.split(f'st{key}')[1]
                identifiers.append(int(identifier))
            coin = max(identifiers)+1
        
        #now we can name the subtasks
        for collocation in collocation_doc_freq[key].keys():
            if collocation not in subtasks[key]: #if it doesn't exist then add it, otherwise it already exists
                subtask_name = f'st{key}{coin}'
                subtasks[key][collocation] = subtask_name
            
                coin+=1 #only increment coin if we add a new subtask
    
    return subtasks, tokenized_runs, participants, raw_runs

#finds the smaller subtaks that are encased within the given subtask (st)
def subtasks_in_subtask(st_dict_list, st):
    #given st as the list form and the subtasks list-dict, return the names of the encased smaller subtasks (not including unigram subtasks)
    if len(st) == 2: #can't encase another subtask inside of a length 2 subtask
        return []
    
    smaller_st_list = [] #will hold the subtasks that  are shorted than the subtask we are checking
    
    
    #get all bigrams in the given subtask, will always be smaller than the given subtask if it gets this far
    st_bigrams = list(nltk.bigrams(st))
    smaller_st_list.extend(list(st_dict_list[2].keys())) #get all subtasks that of size 2
    
    #get the trigrams that exist in the given subtask if the size of the given subtask is greater than 3
    st_trigrams = []
    st_size = len(st)
    if st_size > 3:
        st_trigrams = list(nltk.trigrams(st))
        smaller_st_list.extend(list(st_dict_list[3].keys())) #get all subtasks that of size 3
    
    #TODO if we get bigger sized subtasks, add in a loop that goes through every available size
    
    smaller_st_in_st = [] #will hold encased subtasks
    
    for smaller_st in smaller_st_list: #find the subtasks that are in the given subtask
        if smaller_st in st_bigrams or smaller_st in st_trigrams: #basically we are going through every defined subtask and seeing if it exists in the ngrams we collected
            smaller_st_in_st.append(st_dict_list[len(smaller_st)][smaller_st])
            
    return smaller_st_in_st

    
#=========================================================

#                ENCODING

#=========================================================

#definition of a singel subtask WITHIN a run, this is not the subtask dictionary
class Subtask:
    def __init__(self, name, ngram, start, end, encased_subtasks, encased, left_leads, right_leads):
        self.name = name #string name of the associated ngram
        self.ngram = ngram #the ngram representation of this subtask
        self.start = start
        self.end = end
        self.encased_subtasks = encased_subtasks #the subtasks that are encased in this subtask
        self.encased = encased #boolean for whether this subtask is encased in another one or not
        self.left_leads = left_leads #the subtask that lead into this subtask
        self.right_leads = right_leads #the subtask that this subtask leads into

#definition of a single run object, holding all pertinent data to that run
class Run():
    def __init__(self, participant, task, raw_run, tokenized_run, encoded_run):
        self.participant = participant
        self.task = task
        self.raw_run = raw_run
        self.tokenized_run = tokenized_run
        self.encoded_run = encoded_run

#returns the start and end indexes of a collocation IF it exists in the given tokenized run
def collocation_in_run_indices(collocation, tokenized_run, start_at):
    try: #try to find the collocation
        index = start_at
        first_token = collocation[0] #the first action to find from the given starting point
        while True:
            idx_of_first_token = tokenized_run.index(first_token, index) #this is what fails the try (ValueError, first_token not in list)
            subsequence = tokenized_run[idx_of_first_token:idx_of_first_token+len(collocation)] #get the subsequence of the same size where the starting token is
            if subsequence == collocation: #check if the subsequence is equal to our collocation
                return True, [idx_of_first_token, idx_of_first_token+len(collocation)-1]
            index+=1 #increment to next action if the collocation was not correct (explores entire run, so we can skip actions that are the correct starting token, but not correct collocation)
    except: #if we go through the entire thing without finding it
        return False, None

#checks if two subtasks our overlapping within a run and returns their lead
def is_overlapping(subtask1, subtask2):
    #return 0 if no overlap
    #return 1 if subtask1 leads into subtask2
    #return 2 if subtask2 leads into subtask1
    
    if subtask1.start < subtask2.start and subtask1.end < subtask2.end and subtask1.end >= subtask2.start:
        return 1
    if subtask2.start < subtask1.start and subtask2.end < subtask1.end and subtask2.end >= subtask1.start:
        return 2
    return 0

#checks if a subtask is encased in another
def is_encased(subtask1, subtask2):
    #0 if not encased
    #1 if subtask1 is encased in subtask2
    if subtask1.start >= subtask2.start and subtask1.end <= subtask2.end:
        return 1
    return 0

#create a print method for printing these out in a readable way
def print_encoding(run):
    encoded_string = ''
    for i in range(len(run.tokenized_run)):
        for ngram_size in run.encoded_run.keys():
            for subtask in run.encoded_run[ngram_size]:
                if subtask.start == i and subtask.encased == False:
                    encoded_string += str(subtask.name)
                    if len(subtask.right_leads) > 0:
                        encoded_string += ' ->'
                    encoded_string += ' '
    print(encoded_string)

#return top level encoding
def get_encoding(run):
    top_lvl_encoding = []
    for i in range(len(run.tokenized_run)):
        for ngram_size in run.encoded_run.keys():
            for subtask in run.encoded_run[ngram_size]:
                if subtask.start == i and subtask.encased == False:
                    top_lvl_encoding.append(str(subtask.name))
                    if len(subtask.right_leads) > 0:
                        top_lvl_encoding.append('->')
    return top_lvl_encoding

# steps for hierarchical encoding
# 1. find all subtasks in run (with indices), create subtask objects for them
# 2. find encased subtasks
# 3. find overlapping subtasks and update leads
# 4. create a print method for printing these out in a readable way

def encode_run(tokenized_run, subtasks):
    #find all subtasks in run (with indices), create subtask objects for them
    subtasks_in_run = dict()
    subtasks_in_run[1] = []
    for i, token in enumerate(tokenized_run):
         subtasks_in_run[1].append(Subtask(token, token, i, i, [], False, [], []))
    for key in subtasks.keys():
        subtasks_in_run[key] = []
        for collocation in subtasks[key].keys():
            coin = 0
            while True:
                in_run, indices = collocation_in_run_indices(list(collocation), tokenized_run, coin)
                if in_run:
                    start = indices[0]
                    end = indices[1]
                    found_subtask = Subtask(subtasks[key][collocation], collocation, start, end, [], False, [], [])
                    subtasks_in_run[key].append(found_subtask)
                    coin = start+1
                else:
                    break

    
    #find encased subtasks and store them in encasing subtasks
    ngram_sizes = np.sort(list(subtasks_in_run.keys()))
    for i, ngram_size in enumerate(ngram_sizes[0:-1]):
        ngram_size_subtasks = subtasks_in_run[ngram_size]
        for j, subtask_in_run in enumerate(ngram_size_subtasks):
            for higher_ngram_size in ngram_sizes[i+1:]:
                higher_ngram_size_subtasks_in_run = subtasks_in_run[higher_ngram_size]
                for k, higher_ngram_size_subtask_in_run in enumerate(higher_ngram_size_subtasks_in_run):
                    if is_encased(subtask_in_run, higher_ngram_size_subtask_in_run): #it's inside!
                        
                        higher_ngram_size_subtask_in_run.encased_subtasks.append([ngram_size, j])
                        subtask_in_run.encased = True

            
    #find overlapping subtasks and update leads
    for i, ngram_size in enumerate(ngram_sizes):
        ngram_size_subtasks = subtasks_in_run[ngram_size]
        
        for j, ngram_size_subtask in enumerate(ngram_size_subtasks):
            #go through all in this ngram
            for k, other_ngram_size_subtask in enumerate(ngram_size_subtasks[j+1:]):
                overlapping = is_overlapping(ngram_size_subtask, other_ngram_size_subtask)
                if overlapping == 1: #st1 leads into st2
                    ngram_size_subtask.right_leads.append([ngram_size,j+1+k])
                    other_ngram_size_subtask.left_leads.append([ngram_size,j])
                elif overlapping == 2: #st2 leads into st1
                    ngram_size_subtask.left_leads.append([ngram_size,j+1+k])
                    other_ngram_size_subtask.right_leads.append([ngram_size,j])
            #go through all in higher ngrams
            for k, higher_ngram_size in enumerate(ngram_sizes[i+1:]):
                higher_ngram_size_subtasks = subtasks_in_run[higher_ngram_size]
                for l, higher_ngram_size_subtask in enumerate(higher_ngram_size_subtasks):
                    overlapping = is_overlapping(ngram_size_subtask, higher_ngram_size_subtask)
                    if overlapping == 1: #st1 leads into st2
                        ngram_size_subtask.right_leads.append([higher_ngram_size,l])
                        higher_ngram_size_subtask.left_leads.append([ngram_size,j])
                    elif overlapping == 2: #st2 leads into st1
                        ngram_size_subtask.left_leads.append([higher_ngram_size,l])
                        higher_ngram_size_subtask.right_leads.append([ngram_size,j])
            
    return subtasks_in_run 
        

#for every run, encodes the run and creates a run object for it
def encode_runs(chosen_task, subtasks, tokenized_runs, participants, raw_runs, verbose=False):
    runs = []
    for i, tokenized_run in enumerate(tokenized_runs):
        participant = participants[i]
        raw_run = raw_runs[i]
        new_run = Run(participant, chosen_task, raw_run, tokenized_run, encode_run(tokenized_run, subtasks))
        if verbose:
            print_encoding(new_run)
        runs.append(new_run)
    
    return runs



#=========================================================

#                SIDE EFFECTS

#=========================================================

#manually defines the side effects of all actions
def get_action_side_effects():
    
    #the user... {insert side effects}
    action_side_effects = {
        'ftp': 'access gain to online network protocol',
        'hydra': 'access gain to online network protocol',
        'man': 'information gain of some command',
        'wc' : 'information gain of file(s)',
        'python' : 'python script run',
        'python3' : 'python script run',
        'sshpass' : 'access gain to online network protocol',
        'ssh' : 'access gain to online network protocol',
        'chmod' : 'access mode of a file changed',
        'mv' : 'directories or files renamed or moved',
        'exit' : 'shell exited',
        'nmap' : 'network scan performed',
        'ffuf' : 'hidden files or directories found',
        'wordlist' : 'random list of words generated',
        'cp' : 'file(s) copied',
        'jo' : 'json object created',
        'john' : 'access gain to offline network protocol',
        'hash' : 'hashed path-name list changed',
        'hashcat' : 'access gain to online network protocol',
        'dirb' : 'hidden files or directories found',
        'help' : 'information gain of built-in shell commands',
        'brutespray' : 'access gain to online network protocol',
        'gin' : 'GIN data repositories modified',
        'for' : 'commands ran for each file within a set of files',
        'curl' : 'data transferred',
        'find' : 'file or directory found',
        'pwd' : 'information gain on current directory',
        'mkdir' : 'new directory created',
        'wget' : 'file(s) downloaded from server to local directory',
        'gzip' : 'files compressed',
        'gunzip' : 'gzip compressed file(s) decompressed',
        'head' : 'information gain on the first n specified lines of a file',
        'grep' : 'keywords searched within a file(s)',
        'msfconsole' : 'Metasploit Framework console opened',
        'nc' : 'data written or read between network computers',
        'ip' : 'network configuration modified or viewed',
        'gcc' : 'C or C++ file compiled',
        'id' : 'information gain on user',
        'gobuster' : 'hidden files or directories found',
        'ping' : 'information gain on IP connectivity',
        'dirbuster' : 'hidden files or directories found',
        'read' : 'user input requested',
        'touch' : 'file(s) created or updated',
        'sed' : 'text in a file(s) modified or viewed',
        'ld' : 'object files, archives, or import files combined into single object file',
        'history' : 'information gain on history of command line input',
        'dig' : 'information gain on queried DNS name servers',
        'nslookup' : 'information gain on DNS records',
        'base64' : 'base64 encoding decoded',
        'test' : 'information gain on the validity of file, directory, or command expression',
        'fg' : 'background job brought into foreground',
        'ifconfig' : 'network interface configuration modified or viewed',
        'setxkbmap' : 'keyboard layout mapped to different layout',
        'burpsuite' : 'burpsuite GUI opened',
        'crunch' : 'word list  generated',
        'exploitdb' : 'information gain on ExploitDB installation',
        'searchsploit' :  'information gain of possible attack to perform',
        'ss' : 'information gain of network socket information',
        'wfuzz' : 'common vulnerabilities found in a web application',
        'which' : 'information gain on the location of an executable file',
        'cut' : 'sections of a file extracted',
        'cewl' : 'word list generated',
        'git' : 'git commands run',
        'apt' : 'linux packages installed, upgraded, or updated',
        'whoami' : 'information gain on current user details',
        'apropos' : 'command found',
        'locate' : 'file or directory located',
        'strings' : 'printable strings in a file located',
        'f' : 'information gain on logged-in users',
        'nikto' : 'web server scanned',
        'tar' : 'archive files created, modified, or extracted',
        'pass' : 'password added, edited, generated, or retrieved',
        'diff' : 'information gain on differences between two files',
        'pkill' : 'signal sent to a currently running processes',
        'logout' : 'shell session logged out',
        'sqlite3' : 'sqlite terminal opened',
        'ps' : 'information gain on currently running processes',
        'netstat' : 'information gain on active connections',
        'docker' : 'docker container ran, saved, or modified',
        'crackmapexec' : 'access gain to online network protocol',
        'pw' : 'users or groups added, removed, or modified',
        'ncrack' : 'access gain to online network protocol',
        'sqlmap' : 'penetration testing of server',
        'pip' : 'python packages installed or modified',
        'sudo' : 'command ran as a different user',
        'su' : 'switched to substitute user',
        'clear' : 'command line text cleared',
        'cd' : 'directories switched',
        'ls' : 'information gain on subdirectories and files in directory',
        'll' : 'information gain on subdirectories and files in directory',
        'l' : 'information gain on subdirectories and files in directory',
        'dir' : 'information gain on subdirectories and files in directory',
        'sh' : 'new shell invoked',
        'cat' : 'file opened or modified',
        'vim' : 'file opened or modified',
        'nvim' : 'file opened or modified',
        'nano' : 'file opened or modified',
        'mousepad' : 'file opened or modified',
        'less' : 'information gain of a file\'s contents',
        'echo' : 'user defined expression echoed',
        'nm' : '', #new with ekoparty
        '.sh' : '',
        'ca' : '',
        'whatweb' : '',
        'tmux' : '',
        'openssl' : '',
        'tail' : '',
        'scp' : '',
        'updatedb' : '',
        'plocate' : '',
        'sftp' : '',
        'tldr' : '',
        'make' : '',
        'smbd' : '',
        '7z' : '',
        'nuclei' : '',
        'rb' : '',
        'smb2-quota' : '',
        'smbinfo' : '',
        'smbmap' : '',
        'whereis' : '',
        'enum4linux' : '',
        'telnet' : '',
        'tree' : '',
        'smbclient'  : '',
        'at' : '',
        'gedit' : '',
        'w': '',
        'sm' : '',
        'wordlists' : '',
        'route' : '',
        'cls' : 'clear the command line screen',
        'htop' : '',
        'top' : '',
        'medusa' : '',
        'tcpdump' : '',
        'ruby' : '',
        'seq' : '',
        'date' : '',
        'tnftp' : '',
        
    }
    
    
    return action_side_effects

#returns the side effects of a subtask
def get_subtask_side_effects(subtask_name, subtasks, action_side_effects):
    ngram_len = int(subtask_name.split('st')[1][0]) #find the length of the subtask
    ngram_len_subtasks = subtasks[ngram_len] #get the subtasks of found length
    #find the subtask ngram
    ngram = ()
    for key, val in ngram_len_subtasks.items():
        if val == subtask_name: 
            ngram = key
    #find the side effects associated with the incidivual action in the ngram
    subtask_side_effects = []
    for action in ngram:
        if action in action_side_effects.keys():
            subtask_side_effects.append(action_side_effects[action])
        else:
            subtask_side_effects.append(f'{action}-NA')
    
    return subtask_side_effects

#manually defines the descriptions of all actions
def get_action_descriptions():
    
    #the user attempted to... {insert side effects}
    action_descriptions = {
        'ftp': 'send an unencrypted data communication',
        'hydra': 'use parallelized brute force password cracking on an online network protocol',
        'man': 'read the manual of some command',
        'wc' : 'check a file(s) word count, number of lines, bytes, and characters',
        'python' : 'run a python script or open a python shell',
        'python3' : 'run a python script or open a python shell',
        'sshpass' : 'run ssh in keyboard-interactive mode with an automated password',
        'ssh' : 'send an encrypted data communication',
        'chmod' : 'change the access mode of a file (read, write, execute permissions)',
        'mv' : 'rename or move directories or files',
        'exit' : 'exit a running shell',
        'nmap' : 'perform a network scan',
        'ffuf' : 'use a brute force hidden file or directory search',
        'wordlist' : 'output a random list of words, likely to generate passwords',
        'cp' : 'copy a file(s)',
        'jo' : 'create a command line json object from some input',
        'john' : 'perform an offline brute force password crack',
        'hash' : 'view or change the hashed command path-name list',
        'hashcat' : 'perform a brute force password crack using hashes of previous knowledge',
        'dirb' : 'perform an online hidden directory and file scan with a dictionary based attack',
        'help' : 'display information about built-in shell commands',
        'brutespray' : 'perform a brute force password crack, using output from nmap or a similar output',
        'gin' : 'manage data repositories, using a GIN data management service',
        'for' : 'run a specified command for each file within a set of files',
        'curl' : 'transfer data with urls',
        'find' : 'search for a file or directory',
        'pwd' : 'write the full path name of their current directory',
        'mkdir' : 'create a new directory(ies)',
        'wget' : 'download a file(s) from a server to their local directory',
        'gzip' : 'compress single files',
        'gunzip' : 'decompress a gzip compressed file(s)',
        'head' : 'print the first n specified lines of a file',
        'grep' : 'keyword search within a file(s)',
        'msfconsole' : 'open a console to the Metasploit Framework',
        'nc' : 'read or write data between network computers',
        'ip' : 'modify or view a network configuration',
        'gcc' : 'manually compile a C or C++ file',
        'id' : 'display a user or group name with their numeric id, given the specified user or current user',
        'gobuster' : 'perform an aggressive brute force search of hidden directories and files',
        'ping' : 'test IP connectivity, reachability, and name resolution',
        'dirbuster' : 'perform a parallelized brute force search of hidden directories and files',
        'read' : 'ask for user input, often used in combination with other commands',
        'touch' : 'create or update a file(s)',
        'sed' : 'search, process, or manipulate text in a file(s)',
        'ld' : 'combine object files, archives, or import files into one output object file',
        'history' : 'display the history of the command line input',
        'dig' : 'perform DNS lookups and displays the answers from queried name servers',
        'nslookup' : 'obtain DNS records',
        'base64' : 'decode a base64 encoding',
        'test' : 'check the validity of file, directory, or command expression',
        'fg' : 'bring a background job into the foreground',
        'ifconfig' : 'assign an address to a network interface or display the current network interface configuration',
        'setxkbmap' : 'map their keyboard layout to specified layout',
        'burpsuite' : 'open the burpsuite GUI',
        'crunch' : 'generate a word list from character list',
        'exploitdb' : 'checked if ExploitDB was installed',
        'searchsploit' :  'perform an offline search through exploitdb',
        'ss' : 'display network socket information',
        'wfuzz' : 'use a fuzz method to find common vulnerabilities in a web application',
        'which' : 'identify the location of an executable file associated with a specified command',
        'cut' : 'extract specific sections of a specified file',
        'cewl' : 'crawl over a page or file to create a word list, typically used with something like john the ripper',
        'git' : 'access git commands from the command line',
        'apt' : 'install, upgrade, or update linux packages',
        'whoami' : 'display the current user and some of the user details',
        'apropos' : 'perform a keyword search to find a specific command',
        'locate' : 'locate a file or directory',
        'strings' : 'locate printable strings in a file',
        'f' : 'display logged-in user information',
        'nikto' : 'scan a web server',
        'tar' : 'create, modify, or extract archive files',
        'pass' : 'add, edit, generate, or retrieve a password',
        'diff' : 'display the differences between two files',
        'pkill' : 'send a signal to a currently running processes',
        'logout' : 'logout of the current shell session',
        'sqlite3' : 'open the sqlite terminal to run sql statements from the command line',
        'ps' : 'display information on currently running processes',
        'netstat' : 'display statistics for active connections',
        'docker' : 'run, save, or modify a docker container or list statistics for docker',
        'crackmapexec' : 'utilize the “swiss army knife” of hacking tools, which could include password cracking, remote code execution, etc',
        'pw' : 'to add, remove, or modify users and groups',
        'ncrack' : 'perform authentication cracking',
        'sqlmap' : 'perform penetration testing for SQL injections',
        'pip' : 'install or modify python packages',
        'sudo' : 'run a command as a different user',
        'su' : 'switch to a substitute user',
        'clear' : 'clear the command line text',
        'cd' : 'switch directories',
        'ls' : 'list subdirectories and files in their directory',
        'll' : 'list subdirectories and files in their directory, using the alias for ls -l',
        'l' : 'list subdirectories and files in their directory, using the alias for ls -CF',
        'dir' : 'list subdirectories and files in a directory',
        'sh' : 'invoke a shell',
        'cat' : 'open/edit a file',
        'vim' : 'open/edit a file',
        'nvim' : 'open/edit a file',
        'nano' : 'open/edit a file',
        'mousepad' : 'open/edit a file',
        'less' : 'read a file\'s contents, one screen at a time',
        'echo' : 'echo some user defined expression',
        'nm' : '', #new with ekoparty
        '.sh' : '',
        'ca' : '',
        'whatweb' : '',
        'tmux' : '',
        'openssl' : '',
        'tail' : '',
        'scp' : '',
        'updatedb' : '',
        'plocate' : '',
        'sftp' : '',
        'tldr' : '',
        'make' : '',
        'smbd' : '',
        '7z' : '',
        'nuclei' : '',
        'rb' : '',
        'smb2-quota' : '',
        'smbinfo' : '',
        'smbmap' : '',
        'whereis' : '',
        'enum4linux' : '',
        'telnet' : '',
        'tree' : '',
        'smbclient'  : '',
        'at' : '',
        'gedit' : '',
        'w': '',
        'sm' : '',
        'wordlists' : '',
        'route' : '',
        'cls' : 'clear the command line screen',
        'htop' : '',
        'top' : '',
        'medusa' : '',
        'tcpdump' : '',
        'ruby' : '',
        'seq' : '',
        'date' : '',
        'tnftp' : '',
    }
    
    
    return action_descriptions

#returns the descriptions of a subtask
def get_subtask_description(subtask_name, subtasks, action_descriptions):
    ngram_len = int(subtask_name.split('st')[1][0])#find the length of the subtask
    ngram_len_subtasks = subtasks[ngram_len] #get the subtasks of found length
    #find the subtask ngram
    ngram = ()
    for key, val in ngram_len_subtasks.items():
        if val == subtask_name: 
            ngram = key
    #find the descriptions associated with the incidivual action in the ngram
    subtask_description = []
    for action in ngram:
        if action in action_descriptions.keys():
            subtask_description.append(action_descriptions[action])
        else:
            subtask_description.append(f'{action}-NA')
    
    return subtask_description


#given subtask name and the subtasks dictionary, return the subtask's ngram representation
def get_subtask_ngram(st_name, subtasks):
    ngram_size = int(st_name.split('st')[1][0])
    ngram_subtasks = subtasks[ngram_size]
    ngram_subtasks = {v: k for k, v in ngram_subtasks.items()}
    
    return ngram_subtasks[st_name]

def get_subtasks_ngram(sts, subtasks):
    for st in sts:
        print(f'{st} -> {get_subtask_ngram(st, subtasks)}')