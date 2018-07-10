import math

###############################################################################
# Calculate vector scores of all PRs in the given vector space by tfidf 
#
# Input:
# PRs: all PRs that need to get vector scores
# vectorBase: base of vector space in the training dataset
# vectorBaseCnt: numbers of appearance of each word in vector space
# fileSize: size of training dataset
#
# Output:
# scores: vector scores for all PRs in the input. Same length as PRs in input
#
###############################################################################
def tfidf(PRs, vectorBase, vectorBaseCnt, fileSize):
    vlen = len(vectorBase)
    scores = []
    
    for PR in PRs:
        # for each PR, the vector score is a vector in the same dimensional as vector space. 
        # For each dimension, the value equals to the tfidf value of the word in the PR content, given the whole word dataset
        score = [0.0]*vlen
        
        # actual contents of the PR
        content = PR[1]
        tlen = len(content)
        
        # for each word
        for i in range(tlen):
            word = content[i]
            
            # if this word appears the first time in the PR content
            if (content.index(word) == i):
                try:
                    # Get the correspond index of the word in vector space
                    ind = vectorBase.index(word)
                    # score equals to tfidf value
                    score[ind] = math.log(1+float(content.count(word))/tlen)*math.log(float(fileSize)/vectorBaseCnt[ind])
                    
                # Error happens when the word in content doesn't exist in vector space, which may happen in testing phase
                except ValueError:
                    continue
        
        # append the score into result list
        scores.append(score)
        
    return scores
            