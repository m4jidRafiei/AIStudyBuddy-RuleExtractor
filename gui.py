import pandas as pd
import numpy as np
import statistics
import math

import PySimpleGUI as sg
# for plotting in PySimpleGUI
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import os  # to delete local files
from PIL import Image  # to save DT image

# Process Mining
import pm4py
from pm4py.algo.filtering.dfg import dfg_filtering

# Classification
import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree, _tree
from sklearn.model_selection import KFold

# SMOTE upsampling
import imblearn
from imblearn.over_sampling import SMOTE, SMOTEN

# Evaluation
from sklearn import metrics
from sklearn.metrics import classification_report

# Networks and Visualization
import graphviz
import pydot
import networkx as nx

import collections
from collections import Counter

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore") # ignores all warnings


# PySimleGUI theme
sg.change_look_and_feel('systemdefault')


# --- Explanatory Analysis Modules: ---


def add_course_index(df, feature, is_atomic, pm_index_type):
    """
    adds course-index (or feature name) column to df and index column depending on feature
    
    course-index: 
    - atomic: e.g., course-1_2 with index 2 (for every attempt)
    - non-atomic: e.g., e_course-1 with index 2 (only for passed attempt)
    start_index (only relevant for non-atomic feature): e.g s_course-1
    """
    def order(row, is_atomic):
        s_id = row['studentstudyid']
        c = row['Course']
        sem = row['fachsemester']
        status = row['Final Course Status']

        student = df.loc[df['studentstudyid'] == s_id]
        semesters = sorted(student['fachsemester'].unique())

        idx = semesters.index(sem)

        if is_atomic:
            return c+"_"+str(idx+1), idx+1, None
        else:
            # check if is first attempt (if yes, add start_index)
            if sorted(student.loc[student['Course'] == c]['fachsemester'].values)[0] == sem:
                # if attempt is passed, add end_index 
                if status == "PASSED":
                    return "e_"+c, idx+1, "s_"+c
                else:
                    return None, idx+1, "s_"+c
            else:
                if status == "PASSED":
                    return "e_"+c, idx+1, None
                else:
                    return None, None, None


    def distance(row, is_atomic):
        s_id = row['studentstudyid']
        c = row['Course']
        sem = row['fachsemester']
        status = row['Final Course Status']
        
        student = df.loc[df['studentstudyid'] == s_id]
        semesters = sorted(student['fachsemester'].unique())
        
        first_sem = semesters[0]
        
        if is_atomic:
            return c+"_"+str(sem-first_sem), sem-first_sem, None
        else:
            # check if is first attempt (if yes, add start_index)
            if sorted(student.loc[student['Course'] == c]['fachsemester'].values)[0] == sem:
                # if attempt is passed, add end_index 
                if status == "PASSED":
                    return "e_"+c, sem-first_sem, "s_"+c
                else:
                    return None, sem-first_sem, "s_"+c
            else:
                if status == "PASSED":
                    return "e_"+c, sem-first_sem, None
                else:
                    return None, None, None

    def semester(row, is_atomic):
        s_id = row['studentstudyid']
        c = row['Course']
        sem = row['fachsemester']
        status = row['Final Course Status']

        student = df.loc[df['studentstudyid'] == s_id]
        
        if is_atomic:
            return c+"_"+str(sem), sem, None
        else:
            # check if is first attempt (if yes, add start_index)
            if sorted(student.loc[student['Course'] == c]['fachsemester'].values)[0] == sem:
                # if attempt is passed, add end_index 
                if status == "PASSED":
                    return "e_"+c, sem, "s_"+c
                else:
                    return None, sem, "s_"+c
            else:
                if status == "PASSED":
                    return "e_"+c, sem, None
                else:
                    return None, None, None

    if feature == 'Course-Order':
        df['course-index'], df['index'], df['start-index'] = zip(*df.apply(lambda row: order(row, is_atomic), axis=1))
    elif feature == "Course-Distance":
        df['course-index'], df['index'], df['start-index'] = zip(*df.apply (lambda row: distance(row, is_atomic), axis=1))
    elif feature == "Course-Semester":
        df['course-index'], df['index'], df['start-index'] = zip(*df.apply (lambda row: semester(row, is_atomic), axis=1))
    else: # PM features
        if pm_index_type == 'fachsemester': df['course-index'], df['index'], df['start-index'] = zip(*df.apply (lambda row: semester(row, is_atomic), axis=1))
        if pm_index_type == 'order': df['course-index'], df['index'], df['start-index'] = zip(*df.apply(lambda row: order(row, is_atomic), axis=1))
        if pm_index_type == 'distance': df['course-index'], df['index'], df['start-index'] = zip(*df.apply (lambda row: distance(row, is_atomic), axis=1))

# --- label functions: ---
def get_GPA(row, df, incl_fail):
    s = row['studentstudyid']
    credit_grades = df.loc[(df['studentstudyid'] == s)][['Credit', 'course-grade']].values

    # remove nan grades
    c_g = [[c,g] for c,g in credit_grades if not pd.isna(g)]
    
    # remove 5.0 grades?
    if not incl_fail:
        c_g = [[c,g] for c,g in c_g if g != 5.0]

    weighted_grade = sum([c*g for c,g in c_g])
    total_credits = sum([c for c,g in c_g])

    GPA = 0
    gpa_2 = ''
    gpa_5 = ''

    if total_credits > 0:
        GPA = weighted_grade/total_credits 
        GPA = math.floor(GPA * 10)/10.0  # no rounding, only keep first decimal place
    else: # student just has NaN exam grades 
        GPA = None

    if not pd.isna(GPA):
        gpa_2 = get_2_label('GPA', GPA)
        gpa_5 = get_5_label('GPA', GPA)
    else:
        gpa_2 = None
        gpa_5 = None
    
    return GPA, gpa_2, gpa_5

def compute_exact_GPA(df, GPA_grades):
    incl_fail = True # include also failed grades

    if GPA_grades == 'passed+failed last attempt':
        # keep only last attempt within one semester
        df = df.drop_duplicates(subset=['studentstudyid', 'Course', 'fachsemester'], keep='last')
    elif GPA_grades == 'passed': # onyl passed grades
        incl_fail = False

    df['GPA'], df['GPA-2level'], df['GPA-5level'] = zip(*df.apply(lambda row: get_GPA(row, df, incl_fail), axis=1))
    #df.to_csv('data/test/GPA_GUI.csv', index=False) #added (remove later)
    return df

def get_GPA_label(df, is_binary, GPA_grades):
    
    students = df['studentstudyid'].unique()
    GPAs = []

    df = compute_exact_GPA(df, GPA_grades) # adds GPA column to df

    for s in students:
        if is_binary:
            label_GPA = df.loc[df['studentstudyid'] == s]['GPA-2level'].values[0]
        else:
            label_GPA = df.loc[df['studentstudyid'] == s]['GPA-5level'].values[0]
    
        GPAs.append(label_GPA)
    
    df_y_GPA = pd.DataFrame(np.array(GPAs), columns=['label'])
            
    return df_y_GPA, df

def get_2_label(label, number):
    # label can be 'GPA', 'course-1_2', 'e_course-1', ...

    if number <= 2.5: label += ' <= 2.5'
    elif number > 2.5: label += ' > 2.5'
    else: label = None
    
    return label

def get_5_label(label, number):
    # label can be 'GPA', 'course-1_2', 'e_course-1', ...

    if number <= 1.5: label += ' Excellent'
    elif number <= 2.5: label += ' Good'
    elif number <= 3.5: label += ' Satisfactory'
    elif number <= 4.0: label += ' Sufficient'
    elif number > 4.0: label += ' Failed'
    else: label = None
    
    return label

def get_passfail_label(df, course, is_atomic, label_index, is_pm):
    students = df['studentstudyid'].unique()
    pass_fail = [] 

    if is_atomic:
        for s in students:
            status = list(df.loc[(df['studentstudyid'] == s) & (df['course-index'] == course+'_'+str(label_index))]['Final Course Status'])
            if status:
                pass_fail.append(str(status[0]))
            else:
                pass_fail.append(None)
    else: 
        # we should never come to this case if we change lifecycle (end is when passed)
        label = []

    df_y_passfail_course = pd.DataFrame(np.array(pass_fail), 
                                  columns=['label'])
    
    return df_y_passfail_course

def get_course_grade_label(df, course, is_binary, is_atomic, label_index, is_pm):
    students = df['studentstudyid'].unique()
    grades = []

    i = 0
    j = 0

    for s in students:
        label_index_name = ''
        if is_atomic:
            label_index_name = course+'_'+str(label_index)
            grade = list(df.loc[(df['studentstudyid'] == s) & (df['course-index'] == label_index_name)]['course-grade'])
        else:
            if is_pm:
                label_index_name = 'e_'+course
                grade = list(df.loc[(df['studentstudyid'] == s) & (df['course-index'] == label_index_name)]['course-grade'])
            else:
                label_index_name = 'e_'+course+'_'+str(label_index)
                grade = list(df.loc[(df['studentstudyid'] == s) & (df['course-index'] == 'e_'+course) & (df['index'] == label_index)]['course-grade'])
            
        if grade:
            exact_grade = grade[0]
            if is_binary:
                label_grade = get_2_label(label_index_name, exact_grade)
            else:
                label_grade = get_5_label(label_index_name, exact_grade)
            j += 1
            grades.append(label_grade)
        else:
            i += 1
            grades.append(None)
    
                
    df_y_course = pd.DataFrame(np.array(grades), 
                                  columns=['label'])
    
    return df_y_course
    

def get_label(df, clf_dict):

    label_name = clf_dict['label']

    if label_name == 'Overall GPA':
        is_binary = clf_dict['is_binary_label']
        GPA_grades = clf_dict['GPA_grades']
        label, _ = get_GPA_label(df, is_binary, GPA_grades) 
    else: # course-level label
        course = clf_dict['course']
        is_atomic = clf_dict['is_atomic']
        label_index = clf_dict['label_index']
        is_pm = clf_dict['is_pm']
        if label_name == 'Pass/Fail': # TODO: some courses do have just Passed (course-120 e.g)
            label = get_passfail_label(df, course, is_atomic, label_index, is_pm)
        else: # label_name =='Course grade' # TODO: some courses have only NaN grades (course-120 e.g)
            is_binary = clf_dict['is_binary_label']
            label = get_course_grade_label(df, course, is_binary, is_atomic, label_index, is_pm)

    return label

# ---------------------------------

# --- behavioral feature functions: ---
def get_lifecycle_course_index(df, label_index):
    """
    return start and end feature names (e.g, s_course-1_2, e_course-2_4)
    df: whole df, or student df
    label_index: relevant for course-level label --> only return feature names for 
    attempts that happened before or in parallel to course the label is referring to
    """
    end_indices = df[['course-index', 'index']].values
    start_indices = df[['start-index', 'index']].values

    if pd.isna(label_index):
        all_start_features = list(set([e[0]+'_'+str(int(e[1])) for e in start_indices if pd.notna(e[0])]))
        all_end_features = list(set([e[0]+'_'+str(int(e[1])) for e in end_indices if pd.notna(e[0])]))
    else:
        all_start_features = list(set([e[0]+'_'+str(int(e[1])) for e in start_indices if (pd.notna(e[0]) and e[1] <= int(label_index))]))
        all_end_features = list(set([e[0]+'_'+str(int(e[1])) for e in end_indices if (pd.notna(e[0]) and e[1] <= int(label_index))]))       
    
    all_feature_names = all_start_features + all_end_features
    return all_feature_names

def feature_non_pm(df, clf_dict):

    students = df['studentstudyid'].unique()

    is_atomic = clf_dict['is_atomic']
    label = clf_dict['label']
    label_index = clf_dict['label_index']

    all_feature_names = []
    features = []

    if label == 'Overall GPA': 
        if is_atomic:
            all_feature_names = df['course-index'].unique()
        else:
            all_feature_names = get_lifecycle_course_index(df, None)
    else: # course-level label (only features for courses before or in parallel with course the label is referring to)
        if is_atomic:
            all_feature_names = df.loc[(df['index'] <= label_index)]['course-index'].unique()
        else:
            all_feature_names = get_lifecycle_course_index(df, label_index)

    for s in students:
        l = []
        student_df = df.loc[(df['studentstudyid'] == s)]
        if is_atomic:
            student_features = student_df['course-index'].values # get features for all students (later we will drop students that did not take specific course in case of course-level label)
        else:
            student_features = get_lifecycle_course_index(student_df, None)

        for f_name in all_feature_names: # fill feature values (True, False)
            if f_name in student_features:
                l.append(1) #True
            else:
                l.append(0) #False

        features.append(l)

    df_features = pd.DataFrame(np.array(features), columns=all_feature_names)

    return df_features

def get_all_pm_feature_names(df, clf_dict):
    all_pm_feature_names = []
    courses = df['Course'].unique()

    is_atomic = clf_dict['is_atomic']
    course = clf_dict['course']
    label = clf_dict['label']
    label_index = clf_dict['label_index']

    # dfg nodes are either of form 'course-1_2' (atomic) or 'e_course-1'/'s_course-1' (non-atomic)
    if is_atomic:
        dfg_node_names = df['course-index'].unique() 
        if label == 'Overall GPA':
            right_course = dfg_node_names
        else: # course-level label
            right_course = [course+'_'+str(label_index)]
    else:
        dfg_node_names = []
        for c in courses:
            dfg_node_names.append('s_'+c) 
            dfg_node_names.append('e_'+c)
        if label == 'Overall GPA':
            right_course = dfg_node_names
        else: # course-level label
            right_course = ['e_'+course]

    for left in dfg_node_names:
        for right in right_course:
            if is_atomic: # do not add 'course-X_i->course-Y_j', where i>j
                i = int(left[-1])
                j = int(right[-1])
                if i <= j: 
                    all_pm_feature_names.append(left+'->'+right)
            else:
                all_pm_feature_names.append(left+'->'+right)
    
    return all_pm_feature_names

def student_pm_feature(student_df, clf_dict):
    """
    returns:
    pm_features: list of pm_features_names for a specific student (e.g., e_course-1->s_course-2)
    path_lengths: 
    -- for feature == Path Length: list of path lengths for each feature name 
    -- for Directly/Eventually follows: empty list []
    """
    label = clf_dict['label']
    course = clf_dict['course']
    is_atomic = clf_dict['is_atomic']
    label_index = clf_dict['label_index']
    feature = clf_dict['feature']

    pm_features = []
    path_lengths = []

    if is_atomic:
        node_index_df = student_df[['course-index', 'index']]
    else:
        end_indices = student_df[['course-index', 'index']].dropna() # [[e_course-1, 1], [e_course-2, 3]] only ends
        start_indices = student_df[['start-index', 'index']].dropna() # [[s_course-1, 1], [s_course-2, 1]] only starts
        node_index_df = pd.concat([start_indices.rename(columns={'start-index':'course-index'}), end_indices], axis=0)
        
    all_indices = sorted(node_index_df['index'].unique())
    node_index_list = node_index_df.values

    index = 0 # index of course the label is referring to

    if label == 'Overall GPA':
        right_list = node_index_list
    else: 
        if is_atomic:
            # only features of form course-X_i->course-Y_j if label is refering to course-Y taken in j
            index_list = student_df.loc[student_df['course-index'] == course+'_'+str(label_index)].values
            index = label_index if index_list.size>0 else ''
            right_list = [(course+'_'+str(label_index), index)]
        else:
            # only features of form (s/e)_course-X->e_course-Y if label is refering to course-Y
            index_list = student_df.loc[student_df['course-index'] == 'e_'+course]['index'].values
            index = index_list[0] if index_list.size>0 else ''
            right_list = [('e_'+course, index)]

    if index == '': # student did not take course
        return [], []

    for left, index1 in node_index_list:
        for right, index2 in right_list:
            if feature == 'Path Length':
                if index1 <= index2:
                    path_len = all_indices.index(index2) - all_indices.index(index1)
                    pm_features.append(left+'->'+right)
                    path_lengths.append(path_len)
            elif feature == 'Directly Follows':
                if index1+1 == index2:
                    pm_features.append(left+'->'+right)
            else: # Eventually Follows    
                if index1 < index2:
                    pm_features.append(left+'->'+right)

                
    return pm_features, path_lengths

def feature_pm(df, clf_dict):
    students = df['studentstudyid'].unique()
    all_feature_names = get_all_pm_feature_names(df, clf_dict)
    feature = clf_dict['feature']

    features = []

    for s in students:
        l = []
        student_df = df.loc[(df['studentstudyid'] == s)]
        student_features, path_lens = student_pm_feature(student_df, clf_dict)

        for f_name in all_feature_names: # fill feature values
            if feature == 'Path Length': #numerical values
                if f_name in student_features:
                    idx = student_features.index(f_name)
                    path_length = path_lens[idx]
                    l.append(path_length)
                else:
                    l.append(-1)
            else: # binary values
                if f_name in student_features:
                    l.append(1)
                else:
                    l.append(0)

        features.append(l)

    df_features = pd.DataFrame(np.array(features), 
                                  columns=all_feature_names)
    return df_features

def get_behavioral_features(df, clf_dict):

    feature = clf_dict['feature']
    is_atomic = clf_dict['is_atomic']

    if feature in ['Course-Order', 'Course-Semester', 'Course-Distance']:
        df_behav_feature = feature_non_pm(df, clf_dict)
        
    elif feature in ['Path Length', 'Directly Follows', 'Eventually Follows']:
        df_behav_feature = feature_pm(df, clf_dict) 
    
    #df_behav_feature.to_csv('data/test/test.csv', index=False) #added (remove later)
    return df_behav_feature

# ----------------------------------

# --- survival bias features: ----
def get_exams_per_semester(df, s, max_sem):
    # return list with index describing  fachsemester and value describing number of exams in that fachsemester
    # [1, 3, 2] means 1 exam in fachsemester 1, 3 exams in fachsemester 2,...
    exams_semester = []
    exams_semester_dict = dict(df.loc[(df['studentstudyid'] == s)][['fachsemester']].value_counts())
    
    for i in range(1,max_sem+1):
        exams_semester.append(exams_semester_dict[(i,)] if (i,) in exams_semester_dict.keys() else 0)
        
    return exams_semester

def get_numerical_features(df, clf_dict):
    """
    returns numerical behavioral features: number of total exams, 
    median exams per semester, number of non zero semester
    """
    max_sem = max(df['fachsemester'].values)
    students = df['studentstudyid'].unique()

    label = clf_dict['label']

    if label == 'Overall GPA': # study-level Label
        numerical = [['exams', 'med_exams_per_sem', 'non_zero_sems']]
        for s in students:
            exams_semester = get_exams_per_semester(df, s, max_sem)
            exams = sum(exams_semester) #passed+failed exams
            med_exams_per_sem = int(np.median(exams_semester)) # Median number of exams written per semester
            non_zero_sems = np.count_nonzero(exams_semester) # Number of semesters with exams written 
            l = [exams, med_exams_per_sem, non_zero_sems]
            numerical.append(l)
    else: # course-level label --> only consider number of exams written in parallel with course the label is referring to
        course = clf_dict['course']
        label_index = clf_dict['label_index']
        is_atomic = clf_dict['is_atomic']
        is_pm = clf_dict['is_pm']
        
        numerical = [['exams']]

        for s in students:
            if not is_atomic and is_pm: # label_index not defined in this case
                # get index of 'e_course' (can be index column or fachsemester column)
                label_index = df.loc[(df['studentstudyid'] == s) & (df['course-index'] == "e_"+course)]['index'].values
                if label_index.size > 0:
                    label_index = label_index[0]
                else:
                    label_index = '' # will always lead to exam = 0 in following
                      
            exams = len(df.loc[(df['studentstudyid'] == s) & (df['index'] == label_index)])
            l = [exams]
            numerical.append(l)

    df_numerical = pd.DataFrame(np.array(numerical[1:]), columns=numerical[0])
    
    return df_numerical

def get_course_grade_dict(df, courses):
    '''
    df: exam attempts df
    courses: list of all unique course-IDs in df
    returns dictionary with a list of all grades of a course, and the median grade:
    {
    'course-1': {'grades': [...], 'median-grade': x},
    'course-2': ...
    }
    '''
    course_dict = {}
    for c in courses:
        grades = df.loc[df['Course'] == c]['course-grade'].values
        grades = np.asarray([g for g in grades if ~np.isnan(g)]) # remove nan values (some courses do not provide a grade)
        course_dict[c] = {}
        course_dict[c]['grades'] = grades
        try:
            course_dict[c]['median-grade'] = statistics.median(grades)
        except: # if grades were all nan, the grade list is empty
            course_dict[c]['median-grade'] = np.nan

    """
    For pass/fail courses, we will only see grade values "nan" and "5.0". 
    After removing all nan grades, we are left with 5.0s. It would be wrong to infer a median grade of 5.0 here.
    In those cases, we set the median grade to nan.
    """
    # after removing nan, course-15 has just 5.0s, so median grade would be 5.0      
    course_dict['course-15']['median-grade'] = np.nan

    return course_dict

def get_course_difficulties(course_dict, courses):
    """
    course_dicts: dictionary of course-IDs with grades and median grades for each course
    courses: list of all course-IDs
    returns 4 lists of courses depending on difficulty level
    """
    very_easy = []
    easy = []
    difficult = []
    very_difficult = []

    for c in courses:
        g = course_dict[c]['median-grade']
        if g <= 1.5:
            very_easy.append(c)
        elif 1.6 <= g <= 2.5:
            easy.append(c)
        elif 2.6 <= g <= 3.5:
            difficult.append(c)
        elif 3.6 <= g <= 4.0:
            very_difficult.append(c)

    return very_easy, easy, difficult, very_difficult

def get_difficulty_features(df, clf_dict):
    courses = df['Course'].unique()
    students = df['studentstudyid'].unique()

    course_dict = get_course_grade_dict(df, courses)
    very_easy, easy, difficult, very_difficult = get_course_difficulties(course_dict, courses)

    difficulty_levels_columns = ['very easy exams', 'easy exams', 'difficult exams', 'very difficult exams']
    difficulty_levels = []

    label = clf_dict['label']

    if label == 'Overall GPA': # study-level Label
        for s in students:
            l = [0,0,0,0]
            s_courses = df.loc[df['studentstudyid'] == s]['Course'].values
            # count number of courses in each difficulty level for specific student
            for c in s_courses:
                if c in very_easy: l[0] += 1
                elif c in easy: l[1] += 1
                elif c in difficult: l[2] += 1
                else: l[3] += 1 # c in very_difficult
            difficulty_levels.append(l)
    else: # course-level label --> need to adjust features. Only consider courses that were taken in parallel with the course the label is referring to.
        course = clf_dict['course']
        label_index = clf_dict['label_index']
        is_atomic = clf_dict['is_atomic']
        is_pm = clf_dict['is_pm']

        for s in students:
            if not is_atomic and is_pm: # label_index not defined in this case
                # get index of 'e_course' (can be index column or fachsemester column)
                label_index = df.loc[(df['studentstudyid'] == s) & (df['course-index'] == "e_"+course)]['index'].values
                if label_index.size > 0:
                    label_index = label_index[0]
                else:
                    label_index = '' # will always return an empty s_courses list in following
                   
            l = [0,0,0,0]
            s_courses = df.loc[(df['studentstudyid'] == s) & (df['index'] == label_index)]['Course'].values
            for c in s_courses: 
                if c != course: # don't add difficulty level of course we want to predict a label to 
                    if c in very_easy: l[0] += 1
                    elif c in easy: l[1] += 1
                    elif c in difficult: l[2] += 1
                    else: l[3] += 1 
            difficulty_levels.append(l)
            
    df_difficulty_levels = pd.DataFrame(np.array(difficulty_levels), columns=difficulty_levels_columns)
    
    return df_difficulty_levels

def get_initial_features(df):
    
    df_initial_attr = df[['studentstudyid', 'gender', 'ecc', 'ecg', 'nationality']].drop_duplicates(ignore_index=True)
    df_initial_attr.drop(columns=['studentstudyid'], inplace=True)

    # replacing categorical features to numerical 
    df_initial_attr['gender'].replace(['gender-1', 'gender-2'], [1, 0], inplace=True)
    df_initial_attr['ecc'].replace(['country-1', 'others'], [1, 0], inplace=True)
    df_initial_attr['nationality'].replace(['Country-1', 'others'], [1, 0], inplace=True)
    
    return df_initial_attr

# ------------------------------

def get_features(df, clf_dict):
    features = []
    combinations = clf_dict['combinations']

    # behav occurs in combinations
    if set(combinations) & set(['behav', 'init+behav', 'diff+behav', 'num+behav', 'init+diff+behav', 'init+num+behav', 'diff+num+behav', 'all']):
        behav_feature = get_behavioral_features(df, clf_dict)
    # init occurs in combinations (inititial attributes)
    if set(combinations) & set(['init', 'init+diff', 'init+num', 'init+behav', 'init+diff+num', 'init+diff+behav', 'init+num+behav', 'all']):
        init_feature = get_initial_features(df)
    # diff occurs in combinations (difficulty level)
    if set(combinations) & set(['diff', 'init+diff', 'diff+num', 'diff+behav', 'init+diff+num', 'init+diff+behav', 'diff+num+behav', 'all']): # diff occurs in combinations
        diff_feature = get_difficulty_features(df, clf_dict)
    # num occurs in combinations (numerical behavioral features)
    if set(combinations) & set(['num', 'init+num', 'diff+num', 'num+behav', 'init+diff+num', 'init+num+behav', 'diff+num+behav', 'all']): 
        num_feature = get_numerical_features(df, clf_dict)
    
    # create list of features
    for comb in combinations:
        if comb == 'init': features.append(init_feature)
        elif comb == 'diff': features.append(diff_feature)
        elif comb == 'behav': features.append(behav_feature)
        elif comb == 'num': features.append(num_feature)
        elif comb == 'init+diff': features.append(pd.concat([init_feature,diff_feature], axis=1))
        elif comb == 'init+num': features.append(pd.concat([init_feature,num_feature], axis=1))
        elif comb == 'init+behav': features.append(pd.concat([init_feature,behav_feature], axis=1))
        elif comb == 'diff+num': features.append(pd.concat([diff_feature, num_feature], axis=1))
        elif comb == 'diff+behav': features.append(pd.concat([diff_feature,behav_feature], axis=1))
        elif comb == 'num+behav': features.append(pd.concat([num_feature,behav_feature], axis=1))
        elif comb == 'init+diff+num': features.append(pd.concat([init_feature,diff_feature, num_feature], axis=1))
        elif comb == 'init+diff+behav': features.append(pd.concat([init_feature,diff_feature, behav_feature], axis=1))
        elif comb == 'init+num+behav': features.append(pd.concat([init_feature,num_feature, behav_feature], axis=1))
        elif comb == 'diff+num+behav': features.append(pd.concat([diff_feature,num_feature, behav_feature], axis=1))
        elif comb == 'all': features.append(pd.concat([init_feature,diff_feature,num_feature,behav_feature], axis=1))
    

    return features

# ---- classification ----
def cross_val(model, X, y, cv, score_dict, combi, max_depth, feature_name):
    """
    apply k-fold cross-validation
    
    input:
    model: name of classififcation model 
    X: features (df)
    y: label (df)
    cv: number of folds to use in k-fold cross-validation
    score_dict: will be filled with evaluation results
    combi: name of the combination in question (e.g. init, diff, init+diff, ...)
    label_values: list of label values
    max_depth: max_depth for decision tree
    
    returns:
    score_dict: dict filled with evaluation results
    clf_all: classifier trained on whole dataset (X,y)

    score_dict = {
    'init': {
        'class': {
            'GPA <= 2.5': {
                'precision': [p1, p2, p3, p4],
                'recall': [r1, r2, r3, r4]
            },
            'GPA > 2.5': {
                'precision': [...],
                'recall': [...]
            }
        },
        'accuracy': [x1, x2, x3, x4],
        },
    'diff': ...
    }
    """

    score_dict[combi] = {}
    score_dict[combi]['class'] = {}
    score_dict[combi]['accuracy'] = []

    X = X.values
    y = y.values

    # SMOTE: uncomment this following part if you want to apply SMOTE up-sampling
    # if combi == 'behav' and feature_name == 'Path Length':
    #     sm = SMOTE(random_state=42, k_neighbors = 3) # SMOTE only for continuous features (Path Length)
    # else:
    #     sm = SMOTEN(random_state=42, k_neighbors = 3) # SMOTEN for categorical features (here: True-> 1, False->0)

    # compute folds
    kfold = KFold(n_splits=cv, shuffle=True, random_state=100)

    # apply cross-validation
    for i, (train, test) in enumerate(kfold.split(X,y)):
        X_train = X[train] 
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]


        if model == "DT": 
            clf  = DecisionTreeClassifier(random_state= 100, max_depth = max_depth, min_samples_leaf = 1, min_samples_split = 2) 
        # else: # Future work: extend to other models
        #     clf = ...

        # fit the model
        clf.fit(X_train, np.ravel(y_train))
        # perform classification task on test data
        y_pred = clf.predict(X_test) # testing set was NOT upsampled with SMOTE (here use X_test)--> test set needs to reflect the real world as much as possible
        # Evaluate
        report = classification_report(y_test, y_pred, output_dict=True)

        # fill score_dict
        acc = report['accuracy']
        score_dict[combi]['accuracy'].append(round(acc,4))
        
        labels = list(report.keys())[:-3] # we just want label keys. Last 3 keys are 'accuracy', 'macro avg', 'weighted avg'

        for lab in labels:
            class_recall = report[lab]['recall']
            class_precision = report[lab]['precision']
            if lab not in score_dict[combi]['class']:
                score_dict[combi]['class'][lab] = {}
                score_dict[combi]['class'][lab]['precision'] = []
                score_dict[combi]['class'][lab]['recall'] = []
            score_dict[combi]['class'][lab]['precision'].append(round(class_precision,4))
            score_dict[combi]['class'][lab]['recall'].append(round(class_recall,4))

    # return also one clf -> needed for decision tree plotting
    # decision tree model is trained on whole data
    #X, y = sm.fit_resample(X, y)  # uncomment if applied SMOTE
    clf_all  = DecisionTreeClassifier(random_state= 100, max_depth = max_depth, min_samples_leaf = 1, min_samples_split = 2)
    clf_all.fit(X, np.ravel(y))

    return score_dict, clf_all

def classify(features, label, clf_dict):
    """
    applies 4-fold cross_val to train and evaluate DT on feature+label combinations
    returns:
    score_dict: dict with evaluation results
    figures: list of graphviz objects of DT
    DT_names: name for each graphviz object in figures (e.g init, diff, init+diff, ..)
    parsed: list of extracted rules dor each DT and additional information for rule weighting
    error_msg: error message to show in GUI
    """
    model = clf_dict['model'] # here just DT
    combinations = clf_dict['combinations']
    max_depth = clf_dict['max_depth']
    feature_name = clf_dict['feature']

    label_values = sorted(label[0]['label'].unique())

    score_dict = {}
    DT_names = []
    figures = []
    parsed = []
    error_msg = ""
    
    for i, feature in enumerate(features):

        X = feature
        y = label[i].astype(str)
        
        if len(X) < 4:
            # error: 4-fold cross-val is not possible with only 4 samples
            error_msg = "4-fold Cross-Validation needs at least 4 samples. Number of samples: "+str(len(X))
            return score_dict, DT_names, figures, parsed, error_msg
        
        # apply 4-fold cross-val and receive evaluation scores and clf trained on whole data (DT image will be shown in GUI)
        score_dict, clf = cross_val(model, X, y, 4, score_dict, combinations[i], max_depth, feature_name)


        if model == "DT":
            DT_dot = prepare_DT(clf, X, label_values)
            # fill figures and name list needed for plotting in GUI
            figures.append(DT_dot)
            DT_name = combinations[i]
            DT_names.append(DT_name)
            # DT parsing
            rules = get_rules(clf, list(X.columns), label_values) # no sorting of rules yet
            parsed.append(rules)
        
        # else: #Future work: extend to other classification models
    
    return score_dict, DT_names, figures, parsed, error_msg

# --- DT functions ---
def is_leaf(inner_tree, index):
    # Check whether node is leaf node
    return (inner_tree.children_left[index] == -1 and 
            inner_tree.children_right[index] == -1)

def prune_index(inner_tree, decisions, index=0):
    # Start pruning from the bottom - if we start from the top, we might miss
    # nodes that become leaves during pruning.
    # Do not use this directly - use prune_duplicate_leaves instead.
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_index(inner_tree, decisions, inner_tree.children_right[index])

    # Prune children if both children are leaves now and make the same decision:     
    if (is_leaf(inner_tree, inner_tree.children_left[index]) and
        is_leaf(inner_tree, inner_tree.children_right[index]) and
        (decisions[index] == decisions[inner_tree.children_left[index]]) and 
        (decisions[index] == decisions[inner_tree.children_right[index]])):
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = -1
        inner_tree.children_right[index] = -1
        inner_tree.feature[index] = -2 # needed to change '.feature' for pruned leaves cause extratced rules will still list them otherwise

def prune_duplicate_leaves(clf):
    """
    remove leaves if both have same class.
    E.g internal node x has two child leaves: node 1 and node 2 and both 
    have class label 'GPA > 2.5'. Then we can remove node 1 and node 2, and
    make node x a leaf node with label 'GPA > 2.5'
    """
    decisions = clf.tree_.value.argmax(axis=2).flatten().tolist() # get decision for each node (what class label to predict)
    # recursive function that removes those 'duplicate nodes' starting from the leaves
    prune_index(clf.tree_, decisions)

def get_leaves(clf):
    # removes list of leave node IDs in DT tree 'clf'

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    
    leaves = []
    
    for node_id in range(n_nodes):
        # If the left and right child of a node is the same we have a leaf
        if children_left[node_id] == children_right[node_id]:
            leaves.append(node_id)
            
    return leaves

def prepare_DT(clf, X, label_values):
    """
    prepares DT for plotting in GUI: 
    - prune unnecessary nodes
    - coloring of nodes
    - remove node information, such as sample_count, gini value, ... 

    returns
    updated_dot: graphviz dot file of DT
    """

    prune_duplicate_leaves(clf)
    leaves = get_leaves(clf)

    # get graphviz vizualization
    dot_data = tree.export_graphviz(clf, feature_names = list(X.columns), class_names = label_values,
            filled=True, rounded=True,  
            out_file=None,
                    )
    graph_source = graphviz.Source(dot_data)
    # Parse the source of the graph
    graph = pydot.graph_from_dot_data(graph_source.source)[0]

    colmap = {name: n for n, name in enumerate(set(label_values))}
    color = ['PaleGreen', 'plum', 'khaki', 'coral','skyblue']

    # Change the label of nodes and do coloring
    for node in graph.get_nodes():
        node_id = node.get_name()
        if node_id == 'node':
            node.set_color("white")
        elif node_id != "edge":
            old_label = node.get_label()
            if old_label != 'None' and node_id !='"\\n"':
                if int(node_id) in leaves: # color leave nodes, and only show class label
                    new_label = old_label.split('class = ')[-1][:-1]
                    node.set_fillcolor(color[colmap[new_label]])
                    node.set_color("black")
                else: # internal nodes are white, and only show decision rule 
                    new_label = old_label.split('\\n', 1)[0][1:]
                    node.set_fillcolor("white")
                    node.set_color("black")
                node.set_label(new_label)

    # Get the updated source code
    updated_source = graph.to_string()
    # Create a new Graphviz graph object with the updated source code
    updated_dot = graphviz.Source(updated_source) 
    updated_dot.format = 'png' # needed to show image in GUI, otherwise stored as 'pdf'

    return updated_dot

def get_rules(tree, feature_names, class_names):
    """
    return list of parsed rules from DT and additional information about rule weighting
    input:
    tree: DT classifier
    feature_names: list of feature names
    class_names: list of label names

    returns:
    rules = [[text_rule1, accuracy, samples_%, samples*accuracy, (sample int, sample total int)], [text_rule2, ..., ..., ..., ..., ], ...]
    """
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    N = tree_.n_node_samples[0] # sample count at root node describes total sample count
    
    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"{name} <= {np.round(threshold, 3)}"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"{name} > {np.round(threshold, 3)}"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    # list of all sample count percentages and accuracies
    all_sample_perc = []
    all_acc = []

    rules = []
    for path in paths:  
        rule = ["", 0, 0, 0, 0]
        # extract decisions in path: e.g., course-1_2<0.5, course-2_3>0.5, ..
        for p in path[:-1]: 
            rule[0] += str(p)
            rule[0] += ", "
        if class_names is None:
            rule[0] += "response: "+str(np.round(path[-1][0][0][0],3))
        # extract predicition in path: e.g., class: GPA > 2.5
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            accuracy = np.round(100.0*classes[l]/np.sum(classes),2)
            rule[0] += f"class: {class_names[l]}"
            rule[1] = accuracy
            all_acc.append(accuracy)
        samples = path[-1][1]
        samples_perc = np.round(100*samples/N,2)
        rule[2] = samples_perc
        all_sample_perc.append(samples_perc)
        rule[4] = (samples, N)
        rule[3] = 0 # fill in later 
        rules += [rule]
        

    #normalize all values to be between 0 and 1, and compute combined measure
    all_acc = np.array(all_acc)
    all_sample_perc = np.array(all_sample_perc)
    acc_norm = (all_acc-np.min(all_acc))/(np.max(all_acc)-np.min(all_acc))
    sample_norm = (all_sample_perc-np.min(all_sample_perc))/(np.max(all_sample_perc)-np.min(all_sample_perc))
    for i, rule in enumerate(rules):
        rule[3] = np.round(((acc_norm[i] + sample_norm[i])/2)*100, 2) #combine both measures

    
    return rules

# --------------

def drop_rows(features, label):
    # drops rows where the corresponding label is not defined (removes row from feature df and label df)
    # removes student (row) that e.g have not taken course-1 in semester 2, or rows where we do not have a GPA, because student just took courses without grades
    new_label = label
    new_features = []

    for feature in features:
        feature = feature[new_label['label'].notnull()]
        new_features.append(feature)

    new_label = new_label[new_label['label'].notnull()]

    return new_features, new_label

# TODO?? 
def smote_upsample(features, label):
    return features, label, ""

def get_features_label(clf_dict):
    """
    return:
    features: list of extracted features (dataframe) for each element in clf_dict['combinations']
    labels: extracted label (dataframe) for each feature 
    error_msg: message to show in GUI in case of an error
    """

    # get dataframe for event log
    df = pd.read_csv(clf_dict['file'])
    df = df.rename(columns={"coursescheduleid": "Course"}) # because pure data has different column name

    feature  = clf_dict['feature']
    is_atomic = clf_dict['is_atomic']
    pm_index_type = clf_dict['index_type']
    add_course_index(df, feature, is_atomic, pm_index_type)

    label = get_label(df, clf_dict)

    # Check: Do we have less than two classes?
    error_msg = ""
    unique_labels = label['label'].dropna().unique()
    count_labels = label['label'].value_counts()
    num_labels = len(unique_labels)
    
    if num_labels < 2:
        if num_labels > 0:
            error_msg = "Classification not helpful:\n\nLabel needs to have more than 1 class. Got "+str(num_labels)+" class instead: "+unique_labels[0]+"\n"+str(count_labels)#+"\n\nNumber of data points: "+str(num_rows)
        else:
            error_msg = "Classification not helpful:\n\nLabel needs to have more than 1 class. Got "+str(num_labels)+" class instead."#\n\nNumber of data points: "+str(num_rows)
        return [],[], error_msg


    features = get_features(df, clf_dict)

    # drop rows that don't have a label (for example do not have a grade for course-1 taken in fachsemester 2)
    features, label = drop_rows(features, label)


    n_features = len(features)
    labels = [label] * n_features

    return features, labels, error_msg

def parse_values(values):
    """
    example values:
    {'expl_file': path to event log (.csv), 
    'label': 'Overall GPA', 
    'course': '', 
    'GPA_grades': 'passed+failed last attempt', 
    '2level_radio': True, 
    '5level_radio': False, 
    'model': 'DT', 
    'DT_depth': '3', 
    'atomic_radio': True, 
    'nonatomic_radio': False, 
    'pm_radio': False, 
    'no_pm_radio': True, 
    'index_type': 'fachsemester', 
    'feature': 'Course-Order', 
    'label_index': '', 
    'combinations': ['behav'], 
    }

    return:
    {'file': path to event log (.csv),
    'label': label model should predict,
    'course': course-ID that course-level label refers to,
    'GPA_grades": how to compute the overall GPA,
    'is_binary_label': whether course-grade or overall GPA labels are binary or use 5 levels,
    'model': classification model to use (here only DT, futur work: other models),
    'max_depth': max_depth of DT,
    'is_atomic': whether selected behavioral feature uses atomic or non-atomic definition,
    'is_pm': is selected behavioral feature a PM feature or not?,
    'index_type': what DFG-index to use (only relevant for PM features),
    'feature': name of selected behavioral feature,
    'label_index': index that course-level label refers to (only relevant for course-level label and (atomic features or non-atomic non-PM features)),
    'combinations': list of features to test (behavioral features + survival bias features)
    }
    """
    clf_dict = {}

    clf_dict['file'] = values['expl_file']
    clf_dict['label'] = values['label']
    clf_dict['course'] = values['course']
    clf_dict['GPA_grades'] = values['GPA_grades']
    clf_dict['is_binary_label'] = values['2level_radio']
    clf_dict['model'] = values['model']
    clf_dict['max_depth'] = int(values['DT_depth'])
    clf_dict['is_atomic'] = values['atomic_radio']
    clf_dict['is_pm'] = values['pm_radio']
    clf_dict['index_type'] = values['index_type']
    clf_dict['feature'] = values['feature']
    clf_dict['label_index'] = values['label_index']
    clf_dict['combinations'] = values['combinations']

    return clf_dict

def submit_handler(values):

    clf_dict = parse_values(values)

    score_dict = {} # evaluation results of 4-fold cross-validation
    DT_names = [] # list of names of different DTs
    figures = [] # list of graphviz objects of trained DTs for each selected feature
    parsed = [] # list of rules (and sorting weights) of each DT
    error_msg = "" # error message to show in GUI in case of an error

    lab_idx = clf_dict['label_index']

    do_all_exp = False
    do_indiv = False
    if do_all_exp:
        # automate eval:
        fs = ['Course-Semester', 'Course-Order', 'Course-Distance', 'Path Length', 'Directly Follows', 'Eventually Follows']
        idx_t = ['fachsemester', 'order', 'distance']

        for f in fs:
            clf_dict['feature'] = f
            if f in ['Course-Semester', 'Course-Order', 'Course-Distance']:
                clf_dict['is_pm'] = False

                features, label, error_msg = get_features_label(clf_dict)

                if features != []:
                    score_dict, DT_names, figures, parsed, error_msg = classify(features, label, clf_dict)

            else:
                clf_dict['is_pm'] = True
                for i in idx_t:
                    clf_dict['index_type'] = i

                    features, label, error_msg = get_features_label(clf_dict)

                    if features != []:
                        score_dict, DT_names, figures, parsed, error_msg = classify(features, label, clf_dict)

    
    elif do_indiv:
        nonpm = ['Course-Semester', 'Course-Order', 'Course-Distance', 'Path Length', 'Directly Follows', 'Eventually Follows']
        idx_t = ['fachsemester', 'order', 'distance']

        for f in nonpm:
            clf_dict['feature'] = f
            if f in ['Course-Semester', 'Course-Order', 'Course-Distance']:
                clf_dict['is_pm'] = False
                if f == 'Course-Distance':
                    clf_dict['label_index'] = lab_idx-1
                else:
                        clf_dict['label_index'] = lab_idx

                features, label, error_msg = get_features_label(clf_dict)

                if features != []:
                    score_dict, DT_names, figures, parsed, error_msg = classify(features, label, clf_dict)

            else:
                clf_dict['is_pm'] = True
                for i in idx_t:
                    clf_dict['index_type'] = i

                    if i == 'distance':
                        clf_dict['label_index'] = lab_idx-1
                    else:
                        clf_dict['label_index'] = lab_idx

                    features, label, error_msg = get_features_label(clf_dict)

                    if features != []:
                        score_dict, DT_names, figures, parsed, error_msg = classify(features, label, clf_dict)

    
    else:
        features, label, error_msg = get_features_label(clf_dict)

        if features != []:
            score_dict, DT_names, figures, parsed, error_msg = classify(features, label, clf_dict)


    return score_dict, DT_names, figures, parsed, error_msg

# --- GUI content change Modules: ---
def show_dfg(file, group_attribute, filter_val):
    # read event log, add 5-level GPA column, and format to apply process mining techniques
    df = pd.read_csv(file)

    is_binary = False # 5-level GPA
    GPA_grades = 'passed+failed last attempt' # how to compute GPA
    _, df_gpa = get_GPA_label(df, is_binary, GPA_grades) 
    
    event_log = pm4py.format_dataframe(df_gpa, case_id='studentstudyid', activity_key='Course', timestamp_key='Time-Start')

    # extract sublog, and discover DFG
    sublog = event_log.loc[event_log['GPA-5level'] == 'GPA '+group_attribute]
    dfg, sa, ea = pm4py.discover_dfg(sublog)
    activities_count = pm4py.get_event_attribute_values(sublog, "concept:name")

    # simplify DFG using chosen path filter percentage
    dfg, sa, ea, activities_count = dfg_filtering.filter_dfg_on_paths_percentage(dfg, sa, ea, activities_count, filter_val, keep_all_activities=True)
    pm4py.view_dfg(dfg, sa, ea)

    return

def dfg(student_df, is_atomic):
    # Create an empty directed graph
    G = nx.DiGraph()

    # List of nodes(activites) and their indices (timestamp)
    if is_atomic:
        acts = student_df[['course-index', 'index']].values.tolist()
    else:
        end_indices = student_df[['course-index', 'index']].dropna() # [[e_course-1, 1], [e_course-2, 3]] only ends
        start_indices = student_df[['start-index', 'index']].dropna() # [[s_course-1, 1], [s_course-2, 1]] only starts
        acts = pd.concat([start_indices.rename(columns={'start-index':'course-index'}), end_indices], axis=0).values.tolist()

    order = sorted(student_df['index'].dropna().unique())
    
    # add start/end nodes
    G.add_node('s', color="lightgrey", style = "filled")
    G.add_node('e', color="lightgrey", style = "filled")

    for act in acts:
        G.add_node(act[0], color="lightblue", style = "filled")

    #Specify edges
    for i in range(len(acts)):
        if acts[i][1] == order[0]: #first nodes
            G.add_edge('s', acts[i][0], color='lightgrey', style='dashed')
        if acts[i][1] == order[-1]: #last nodes
            G.add_edge(acts[i][0], 'e', color='lightgrey', style='dashed')
        for j in range(len(acts)):
            if i != j:
                if order.index(acts[i][1])+1 == order.index(acts[j][1]):
                    G.add_edge(acts[i][0], acts[j][0], color='grey')


    dot = nx.drawing.nx_pydot.to_pydot(G).to_string()

    src = graphviz.Source(dot) # dot is string containing DOT notation of graph

    return G, src

def show_student_dfg(file, student, is_atomic, index_type):
    df = pd.read_csv(file) 
    student_df = df.loc[(df['studentstudyid'] == student)]

    add_course_index(student_df, '', is_atomic, index_type)

    _, src = dfg(student_df, is_atomic)

    src.format = 'png'
    src.view()

    # delete files created by graphviz .view() function
    # os.remove('Source.gv') 
    # os.remove('Source.gv.png') 
    return

def get_table_values(score_dict):
    # prepare evaluation results to show in GUI table

    data = []
    for feature, val in score_dict.items():
        row = []
        row.append(feature)
        acc = np.asarray(val['accuracy'])
        row.append(str(round(acc.mean(),2))+"+/-"+str(round(acc.std(),2)))
        first_row = True
        for label, scores in val['class'].items():
            if first_row:
                row.append(label)
                prec = np.asarray(scores['precision'])
                rec = np.asarray(scores['recall'])
                row.append(str(round(prec.mean(),2))+"+/-"+str(round(prec.std(),2)))
                row.append(str(round(rec.mean(),2))+"+/-"+str(round(rec.std(),2)))
                first_row = False
                data.append(row)
            else:
                row = ["", ""]
                row.append(label)
                prec = np.asarray(scores['precision'])
                rec = np.asarray(scores['recall'])
                row.append(str(round(prec.mean(),2))+"+/-"+str(round(prec.std(),2)))
                row.append(str(round(rec.mean(),2))+"+/-"+str(round(rec.std(),2)))
                data.append(row)
        data.append(['', '', '', '', ''])
    
    return data

def get_possible_indices(values):
    # return possible label indices for specific course and feature

    indices = []

    clf_dict = parse_values(values)

    course = clf_dict['course']
    feature = clf_dict['feature'] 
    pm_index_type = clf_dict['index_type']
    is_atomic = clf_dict['is_atomic']
    file = clf_dict['file']

    df = pd.read_csv(file)

    add_course_index(df, feature, is_atomic, pm_index_type) # add index column to df
    
    # which indices exist for this course?
    indices = sorted(df.loc[(df['Course'] == course)]['index'].dropna().astype(int).unique())

    return indices

def get_student_list(file):
    df = pd.read_csv(file)
    return list(df['studentstudyid'].unique())

def get_course_list(file):
    df = pd.read_csv(file)
    return list(df['Course'].unique())

def sort_rules(parsed, event):
    # parsed = [[rule, accuracy, samples%, accuracy*sample, (sample int, sample total int)]]
    rules = parsed

    # sort rules
    if event == "sort_samples":
        rules = sorted(rules, key = lambda x: x[2], reverse=True)
        rules = [r[0]+' ('+str(r[2])+'%, '+str(r[4][0])+' samples out of '+str(r[4][1])+')' for r in rules]
    elif event == "sort_accuracy":
        rules = sorted(rules, key = lambda x: x[1], reverse=True)
        rules = [r[0]+' ('+str(r[1])+'%)' for r in rules]
    elif event == "sort_both":
        rules = sorted(rules, key = lambda x: x[3], reverse=True)
        rules = [r[0]+' ('+str(r[3])+')' for r in rules]
    else:
        rules = [r[0] for r in rules]

    return '\n'.join(rules)

# --- GUI windows change Modules: ---
def collapse(layout, key, visible):
    """
    Helper function that creates a Column that can be later made hidden, thus appearing "collapsed"
    :param layout: The layout for the section
    :param key: Key used to make this section visible / invisible
    :param visible: visible determines if section is rendered visible or invisible on initialization
    :return: A pinned column that can be placed directly into your layout
    :rtype: sg.pin
    """
    return sg.pin(sg.Column(layout, key=key, visible=visible, pad=(0,0)))

def clear_gui(window, combinations):
    # empties the table
    window['table'].update(values=[])

    # make all feature buttons invisible
    for comb in combinations:
        key = comb+"_btn"
        window[key].update(visible=False) 
    return

def popUp(text):
    #opens a popUp window with text

    # define the window layout
    layout = [
        [sg.T(text)]
    ]

    window = sg.Window('', layout)

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
    window.close()

# new window that shows DT image and rules
def open_window(fig, title, parsed=None):
    """
    fig: Graphviz object of trained DT
    title: Name of combination the window is referring to 
    parsed: list of DT rules with rule weighting information
    """

    w,h = sg.Window.get_screen_size()

    fig.render(filename='fig')

    text_colour = "#082567"
    
    # DT image
    left = [
        [sg.T('Decision Tree', text_color=text_colour, font='bold')],
        [sg.Column([[sg.Image('fig.png', key='image')]], size=(w*0.47, h*0.7), scrollable=True, element_justification="center")],    
        [sg.Button('Open Image', button_color=('black', "#FFB557")),
        sg.InputText(key='img-name-in', do_not_clear=False, enable_events=True, visible=False),
        sg.FileSaveAs('Save Image', file_types=(('PNG', '.png'), ('JPG', '.jpg')))],
        [sg.T(key='img-name', text_color=text_colour)],
    ]

    rules = [r[0] for r in parsed]
    rules = '\n'.join(rules)

    # Rules
    right = [
        [sg.T('Rules', text_color=text_colour, font='bold')],
        [sg.Column([[sg.T(rules, key='parse-text')]], size=(w*0.47, h*0.7), scrollable=True, key='column_parse')],
        [
        sg.T('Sort rules:', text_color=text_colour),
        sg.Radio('Default', "sort-radio", default=True, enable_events=True, key="sort_default"), 
        sg.Radio('Sample Count', "sort-radio", default=False, enable_events=True, key="sort_samples"),
        sg.Radio('Accuracy', "sort-radio", default=False, enable_events=True, key="sort_accuracy"),
        sg.Radio('Sample Count & Accuracy', "sort-radio", default=False, enable_events=True, key="sort_both"),
        ],
        [sg.InputText(key='rule-name-in', do_not_clear=False, enable_events=True, visible=False),
        sg.FileSaveAs('Save Rules', file_types=(('CSV', '.csv'),))],
        [sg.T(key='rule-name', text_color=text_colour)],
    ]

    layout = [
        [sg.Column(left, size=(w/2, h*0.9), element_justification="center"),
        sg.VSeperator(),
        sg.Column(right, element_justification="center")],
    ]

    # create the window
    window = sg.Window(title+' model', layout, resizable=True, finalize=True)

    window.Maximize()

    while True:
        event, values = window.read()

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        elif event == "Open Image":
            fig.view()
        
        # save image as .png
        elif event == 'img-name-in':
            filename = values['img-name-in']
            if filename:
                try:
                    image = Image.open('fig.png')
                    image.save(filename)
                    window['img-name'].update('saved as: '+filename)
                except Exception as e: window['img-name'].update(e) # show error message in gui
        
        # save rules as .csv
        elif event == 'rule-name-in':
            filename = values['rule-name-in']
            if filename:
                try:
                    csv_file = open(filename, "w")
                    csv_file.write(rules)
                    csv_file.close()
                    window['rule-name'].update('saved as: '+filename)
                except Exception as e: window['rule-name'].update(e) # show error message in gui
        
        # sort rules
        elif event in ['sort_default', 'sort_samples', 'sort_accuracy', 'sort_both']:
            rules = sort_rules(parsed, event)
            window['parse-text'].update(rules)
            window.refresh()
            window['column_parse'].contents_changed() #update scrollbar


    # delete files created by graphviz .render() function
    os.remove('fig') #fig
    os.remove('fig.png') #fig.png

    window.close()

def gui():

    text_colour = '#082567'
    button_color = ('black', '#FFB557')

    # combinations of feature + survival bias features
    combinations = ['init', 'diff', 'num', 'behav', 'init+diff', 'init+num', 'init+behav', 'diff+num', 'diff+behav', 'num+behav', 'init+diff+num', 'init+diff+behav', 'init+num+behav', 'diff+num+behav', 'all']
    
    # returned lists after click on submit:
    images = []
    figures = []
    parsed = []

    label_type_row = [
                [sg.Text("Label Type:", text_color=text_colour), 
                sg.Radio('2-level', "label-type-radio", default=True, key="2level_radio"),
                sg.Radio('5-level', "label-type-radio", default=False, key="5level_radio")
                ],
            ]
    
    # GUI controls for feature/label extraction
    left_column = [  
                [sg.Text("Select an event log (.csv):", text_color=text_colour)],
                [sg.InputText(key="expl_file"), 
                sg.FileBrowse(file_types=[("CSV Files", "*.csv")])],
                [sg.Text("Label:", text_color=text_colour), 
                sg.Combo(['Overall GPA', 'Pass/Fail', 'Course grade'], default_value='Overall GPA', key='label', enable_events=True),
                sg.Text("for which course?", text_color=text_colour, key='course_text', visible=False), 
                sg.Combo([], default_value='', key='course', visible=False, enable_events=True, size=(12,1)),
                ],
                [sg.Text("Which grades to include?", text_color=text_colour, key='GPA_grades_text'),
                 sg.Combo(['passed+failed last attempt', 'passed+failed both attempts', 'passed'], default_value='passed+failed last attempt', key='GPA_grades', enable_events=True),
                ],
                [collapse(label_type_row, 'label_type_row', True)],
                [sg.Text("Explanation Model:", text_color=text_colour), 
                sg.Combo(['DT'], default_value='DT',key='model', enable_events=True),
                sg.Text("depth:", text_color=text_colour, key="DT_depth_text", visible=True),
                sg.InputText(default_text = "5", key = "DT_depth", size=(5,1), visible = True),
                ],
                [sg.Text("Feature Type:", text_color=text_colour, visible=True, key="type_text"), 
                sg.Radio('Atomic', "type-radio", default=True, enable_events=True, key="atomic_radio", visible=True), 
                sg.Radio('Non-Atomic', "type-radio", default=False, enable_events=True, key="nonatomic_radio", visible=True)
                ],
                [sg.Text("PM feature?", text_color=text_colour,), 
                sg.Radio('Yes', "pm-radio", default=False, enable_events=True, key="pm_radio"), 
                sg.Radio('No', "pm-radio", default=True, enable_events=True, key="no_pm_radio"),
                sg.Text("index type?", text_color=text_colour, key='index_type_text', visible=False), # index in DFG for PM atomic feature
                sg.Combo(['fachsemester', 'order', 'distance'], default_value='fachsemester', key='index_type', visible = False),
                ],
                [sg.Text("Bahavioral Feature:", text_color=text_colour), 
                sg.Combo(['Course-Order', 'Course-Semester', 'Course-Distance'], default_value='Course-Order',key='feature', enable_events=True),
                sg.Text("label index:", text_color=text_colour, key="label_index_text", visible=False),
                sg.Combo([] ,key='label_index', visible = False),
                sg.Button('update', button_color=('black', "#FFB557"), visible = False)
                ],
                [sg.Text("Combinations:", text_color=text_colour),
                sg.Listbox(values = combinations, key='combinations', select_mode = 'multiple', size=(30, 6)),
                ],
                [sg.Button('Submit'), sg.Exit()]
            ]
    
    # Evaluation results in Table
    data = []
    headings = ['Feature', 'Accuracy', 'Class', 'Precision', 'Recall']
    right_column = [
                [sg.Table(values=data, headings=headings, max_col_width=30,
                    auto_size_columns=False,
                    def_col_width=13,
                    justification='center',
                    num_rows=20,
                    key='table',
                    expand_x=False,
                    expand_y=True,
                    vertical_scroll_only=False)],
                ]

    # Feature buttons to see DT image and rules (only visible after submit)
    image_buttons = [[],[]] # two lines of buttons (positioning in GUI)
    for i, comb in enumerate(combinations):
        btn = sg.Button(comb, visible=False, enable_events=True, key=comb+"_btn", button_color=button_color, size=(16,2))
        if i <= 6:
            image_buttons[0].append(btn)
        else:
            image_buttons[1].append(btn)

    explanatory_layout = [
        [sg.Column(left_column),
         sg.VSeperator(),
         sg.Column(right_column)],
        [sg.HorizontalSeparator()],
        [sg.Column(image_buttons)]
    ]

    descriptive_layout = [
        [sg.Text("")],
        [sg.Text("Select an event log (.csv):", text_color=text_colour)],
        [sg.InputText(key="descr_file"),
         sg.FileBrowse(file_types=[("CSV Files", "*.csv")])],
        [sg.Text("Select grouping attribute", text_color=text_colour),
         sg.Combo(['Excellent', 'Good', 'Satisfactory', 'Sufficient',
                   'Failed'], default_value='Excellent', key='descr_group'),
         ],
        [sg.Text("Filter Paths", text_color=text_colour),
         sg.Slider(range=(0, 100), default_value=0, resolution=.1, size=(
             30, 10), orientation='horizontal', key='descr_filter')
         ],
        [sg.Button('Show DFG', key='descr__show')]
    ]

    dfg_layout = [
        [sg.Text("")],
        [sg.Text("Select an event log (.csv):", text_color=text_colour)],
        [sg.InputText(key="dfg_file"),
         sg.FileBrowse(file_types=[("CSV Files", "*.csv")], enable_events=True)],
        [sg.Text("Select student", text_color=text_colour),
         sg.Combo([], key='dfg_student', size=(8, 1)),
         sg.Button('update', key="dfg_get_students",
                   button_color=('black', "#FFB557")),
         ],
        [sg.Radio('Atomic', "dfg-radio", default=True, key="dfg_is_atomic", visible=True, enable_events=True),
         sg.Radio('Non-Atomic', "dfg-radio", default=False,
                  key="dfg_is_nonatomic", visible=True, enable_events=True),
         ],
        [sg.Text("index type:", key='dfg_index_text'),
         sg.Combo(['fachsemester', 'order', 'distance'],
                  default_value='fachsemester', key='dfg_index_type'),
         ],
        [sg.Button('Show DFG', key="dfg_show")]
    ]

    tab_group = [
                [sg.TabGroup(
                    [[
                        sg.Tab('Explanatory Analysis', explanatory_layout),
                        sg.Tab('Descriptive Analysis', descriptive_layout,
                               element_justification='center'),
                        sg.Tab('Student DFGs', dfg_layout,
                               element_justification='center'),
                    ]],
                    tab_location='centertop',
                    selected_background_color="#FFB557"
                )]
    ]

    window = sg.Window("AiStudyBuddy", tab_group,
                       resizable=True, location=(154, 45))

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, 'Exit'):
            break

        # --- descriptive tab events: ---

        # Show discovered DFG
        if event == 'descr__show':
            if values['descr_file'] == "": 
                popUp("Please select a csv file")
            else:
                show_dfg(values['descr_file'], values['descr_group'], values['descr_filter'])

        # --- Student DFG tab events: ---

        # Show index type options only for atomic DFG
        if event == "dfg_is_atomic":
            for key in ['dfg_index_text', 'dfg_index_type']:
                window[key].update(visible=True)
        elif event == "dfg_is_nonatomic":
            for key in ['dfg_index_text', 'dfg_index_type']:
                window[key].update(visible=False)
        
        # Show all student IDs in dropdown
        if event == 'dfg_get_students':
            if values['dfg_file'] == "": 
                popUp("Please select a csv file")
            else:
                student_list = get_student_list(values['dfg_file'])
                window['dfg_student'].update(values=student_list, value=student_list[0])

        # Show student DFG
        if event == "dfg_show":
            if values['dfg_file'] == "": 
                popUp("Please select a csv file")
            else:
                show_student_dfg(values['dfg_file'], values['dfg_student'], values['dfg_is_atomic'], values['dfg_index_type'])
        
        # --- Explanatory Tab events: ---

        # Collape label type row for 'overall GPA' or 'course grade' label
        if event=='label' and values['label'] in ['Course grade', 'Overall GPA']:
                window['label_type_row'].update(visible=True)
        elif event=='label' and values['label'] not in ['Course grade', 'Overall GPA']:
                window['label_type_row'].update(visible=False)

        # remove GPA computation drop down from GUI if course-level label is clicked
        if event=='label' and values['label'] != 'Overall GPA':
                window['GPA_grades_text'].update(visible=False)
                window['GPA_grades'].update(visible=False)
        elif event=='label' and values['label'] == 'Overall GPA':
                window['GPA_grades_text'].update(visible=True)
                window['GPA_grades'].update(visible=True)

        # for course-level label, get course-list, show in dropdown, show label index options
        if event == 'label' and values['label'] in ['Pass/Fail', 'Course grade']:
            if values['expl_file'] == "": 
                popUp("Please select a csv file")
            else:
                course_list = get_course_list(values['expl_file'])
                window['course'].update(values=course_list, value=course_list[0])
                for key in ['course_text', 'course']:
                    window[key].update(visible=True)
                if values['atomic_radio']:
                    for key in ['label_index_text', 'label_index', 'update']:
                        window[key].update(visible=True)
                else:
                    if values['no_pm_radio']:
                        for key in ['label_index_text', 'label_index', 'update']:
                            window[key].update(visible=True)
                    else:
                        for key in ['label_index_text', 'label_index', 'update']:
                            window[key].update(visible=False)            
        elif event == 'label' and values['label'] not in ['Pass/Fail', 'Course grade']:
            for key in ['course_text', 'course', 'label_index_text', 'label_index', 'update']:
                window[key].update(visible=False)

        # Hide label index and DFG index option for non-atomic + PM
        if event == 'nonatomic_radio':
            if (values['label'] in ['Pass/Fail', 'Course grade']) and (values['no_pm_radio']):
                for key in ['label_index_text', 'label_index', 'update']:
                    window[key].update(visible=True)
            else:
                for key in ['label_index_text', 'label_index', 'update']:
                    window[key].update(visible=False)
            if values['pm_radio']:
                window['index_type_text'].update(visible=False)
                window['index_type'].update(visible=False)
        elif event == 'atomic_radio':
            if values['label'] in ['Pass/Fail', 'Course grade']:
                for key in ['label_index_text', 'label_index', 'update']:
                    window[key].update(visible=True)
            if values['pm_radio']:
                window['index_type_text'].update(visible=True)
                window['index_type'].update(visible=True)

        # Change feature dropdown and show DFG index options for PM case
        if event == "pm_radio":
            window['index_type_text'].update(visible=True)
            window['index_type'].update(visible=True)
            window['feature'].update(values=['Path Length', 'Directly Follows', 'Eventually Follows'], value='Path Length')
            if (values['label'] in ['Pass/Fail', 'Course grade']) and (values['atomic_radio']):
                for key in ['label_index_text', 'label_index', 'update']:
                    window[key].update(visible=True)
            else: 
                for key in ['label_index_text', 'label_index', 'update']:
                    window[key].update(visible=False)
            if values['nonatomic_radio']:
                window['index_type_text'].update(visible=False)
                window['index_type'].update(visible=False)

        elif event == "no_pm_radio":
            window['index_type_text'].update(visible=False)
            window['index_type'].update(visible=False)
            no_pm_features = ['Course-Order', 'Course-Semester', 'Course-Distance']
            window['feature'].update(values=no_pm_features, value='Course-Order')
            if values['label'] in ['Pass/Fail', 'Course grade']:
                for key in ['label_index_text', 'label_index', 'update']:
                    window[key].update(visible=True)
            else: 
                for key in ['label_index_text', 'label_index', 'update']:
                    window[key].update(visible=False)
        
        # Show label index options for specific course and feature
        if event == "update":
            indices = get_possible_indices(values)
            window['label_index'].update(values=indices, value=indices[0])

        # User submits
        if event == "Submit":
        
            if values['expl_file'] == "": 
                popUp("Please select a csv file")
            elif values['combinations'] == []:
                popUp("Please select at least one combination")
            else:
                clear_gui(window, combinations)
                score_dict, DT_names, figures, parsed, error_msg = submit_handler(values)

                data = []
                if score_dict != {}:
                    data = get_table_values(score_dict)
                
                window['table'].update(values=data)

                # make feature buttons visible
                for comb in combinations:
                    if comb in DT_names:
                        key = comb+'_btn'
                        window[key].update(visible=True)
                    # else:
                    #     window[key].update(visible=False)    

                if error_msg != "":
                    popUp(error_msg)

        # Click on feature button
        if event in [comb+"_btn" for comb in combinations]: 
            name = event[:-4]
            idx = DT_names.index(name)
            open_window(figures[idx], name, parsed[idx])

    window.close()


if __name__ == "__main__":
    gui()
