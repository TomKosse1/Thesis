import glob
import csv
from collections import OrderedDict
from operator import itemgetter

def countFeature(label, feature, mode):
    """Counts all occurences of a certain Feature and Label, order by value
    Parameter label: 'positive' or 'negative'
    Parameter feature: 'sentiment', 'actors', 'directors', 'genre' or 'titles'
    Parameter mode: 'train', 'test' or 'full'
    Returns dictionary"""
    
    path = 'C:\\Users\\Tom\\Documents\\Informatiekunde\\Thesis\\data\\' + mode + '\\' + label + '\\'
    feature = 'C:\\Users\\Tom\\Documents\\Informatiekunde\\Thesis\\features\\' + feature +'.txt'
    allFiles = glob.glob(path + "*.txt")
    featdict = {}
    for review in allFiles:
        file = open(review, 'r', encoding='utf8').read().lower()
        featreader = csv.reader(open(feature, 'r'), delimiter= '\n')
        for word in featreader:
            if word[0] in file and word[0] not in featdict:
                featdict.update({word[0]: 1})
            elif word[0] in file and word[0] in featdict:
                featdict[word[0]] += 1
    ordered = OrderedDict(sorted(featdict.items(), key=itemgetter(1), reverse=True))

    return ordered

def totalCount(dictionary):
    """Computes total of occurences of a certain Feature
    Parameter dictionary: dictionary returned by countFeature
    Returns integer"""
    
    count = 0
    for key, value in dictionary.items():
        count += value
    
    return count

def removeLowest(dictionary):
    """Removes all occurences of features that occur less than 5 times in total, order by value
    Parameter dictionary: dictionary returned by countFeature
    Returns dictionary"""
    
    new_dict = dict(dictionary)
    for key, value in dictionary.items():
        if value < 5:
            del new_dict[key]
    new_dict = OrderedDict(sorted(new_dict.items(), key=itemgetter(1), reverse=True))
    
    return new_dict

def calcPercentage(dictionary, total):
    """Computes relative occurences of a certain Feature, order by value
    Parameter dictionary: dictionary returned by countFeature
    Parameter total: integer, total number of occurences of an entire feature
    Returns dictionary"""
    
    per_dict = {}
    for key, value in dictionary.items():
        percentage = round((value / total * 100), 2)
        per_dict.update({key: percentage})
    per_dict = OrderedDict(sorted(per_dict.items(), key=itemgetter(1), reverse=True))
    
    return per_dict

def excludeNames(neg_dict, pos_dict):
    """Removes all occurences of Features that appeer in both neg_dict and pos_dict, order by value
    Parameter neg_dict: dictionary thats returned by countFeature with negative label
    Parameter pos_dict: dictionary thats returned by countFeature with positive label
    Returns two dictionaries"""
    
    new_neg_dict = dict(neg_dict)
    new_pos_dict = dict(pos_dict)
    for key, value in neg_dict.items():
        if key in pos_dict:
            del new_neg_dict[key]
            del new_pos_dict[key]
        else:
            continue
    neg_ordered = OrderedDict(sorted(new_neg_dict.items(), key=itemgetter(1), reverse=True))
    pos_ordered = OrderedDict(sorted(new_pos_dict.items(), key=itemgetter(1), reverse=True))
    
    return neg_ordered, pos_ordered