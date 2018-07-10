from author import Author, AuthorList
import expertise
import vectorSpace
import os
import csv, sys
import time

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer

# generate stopwords set as preparation of cleaning PR contents
stop_nltk = set(stopwords.words("english"))
stop_Add = set(["add", "delete", "note", "thank", "another", "please", "per", "test", "implement", "complete", "hello", "fix", "say", "said", "would", "one", "back", "could", "thought", "think", "see", "seem", "want", "like", "still", "go", "went", "around", "make", "made", "come", "came", "hi", "much", "wa", "well", "though", "only", "onli", "might", "away", "even", "know", "many", "good", "get", "got", "right", "must", "great", "us", "something", "yet", "app", "use", "really", "day", "put", "set", "ok"])
stop_sw = set(get_stop_words('en'))
stopW = stop_nltk.union(stop_Add).union(stop_sw)

# Stemming tools. Used for cleaning PR contents
p_stemmer = PorterStemmer()

###############################################################################
# list of all authors in the training dataset of a project. 
# Reset before training each project
# class AuthorList is defined in author.py
###############################################################################
authors = AuthorList()

###############################################################################
# List of scores of all PRs in vector space. Only training dataset included
# Reset before training each project
# vectorScore[i] is a vector score, performing as a list, of the i-th PR
###############################################################################
vectorScore = []

###############################################################################
# Chart of relation scores in common networks. Only training dataset included
# Reset before training each project
# relationScore[i][j] is a relation score from author i to author j
# relationScore is not a symmetery chart. 
# (i.e. relationScore[i][j] == relationScore[j][i] is UNNECESSARY)
###############################################################################
relationScore = []

###############################################################################
# List of all words in the training dataset of a project as the base of vector space.
# Reset before training each project
# Stopwords are not included
###############################################################################
vectorBase = []

###############################################################################
# List of counts of appearance of every word in vectorSpace in the project
# Reset before training each project
# vectorBaseCnt[i] refers to number of times the i-th word in vectorSpace has appeared in the project
###############################################################################
vectorBaseCnt = []

###############################################################################
# List of all PRs in the training dataset of a project. 
# Reset before training each project
# data is loaded from .csv input files
# PR[i] refers to the i-th PR in the training dataset
# for each PR, there are several attributes:
# [0]: "PR" sign  [1]: title  [2]: content  [4]: user_list  [5]: start_time  [6]: end_time
###############################################################################
PRs = []

baseline = 0.0
deadline = 0.0

# function to judge if a statement (content/title/author name) only contains general ASCII characters
def judgeLegal(word):
    return (len(word)>0) and (all (ord(c)<128 for c in word))

# function to judge if a statement (content/title/author name) only contains English letters
def judgeEnglish(word):
    return word.isalpha() and judgeLegal(word)


# Train models from training dataset
def Train(file):
    global PRs, vectorBase, vectorBaseCnt, relationScore, vectorScore, authors, baseline, deadline
    
    # Reset of global variables
    baseline = time.time()
    deadline = 0.0
    csv_file = csv.reader(open(file,'r',errors='ignore'))
    PRs = []
    vectorBase = []
    vectorBaseCnt = []
    authors.clear()
    
    # Read each line (PR) from training dataset
    # PR attributes: [0]: PR  [1]: title  [2]: content  [4]: user_list  [5]: start_time  [6]: end_time
    for e, PR in enumerate(csv_file):
        # if the line is not PR, skip
        if (PR[0] != "PR"):
            continue
        
        # Get title & content of the PR. And merge them.
        contentTitle = word_tokenize(PR[1])
        contentDetail = word_tokenize(PR[2])
        contentTitle.extend(contentDetail)
        content = contentTitle
        
        # Remove stopwords and generate vector space
        cleanContent = []
        # iterate each word
        for i in range(len(content)):
            word = content[i]
            lword = word.lower()
            stword = p_stemmer.stem(lword)
            # if the word is a legal English word and not a stopword, add into the actual content list
            if (judgeEnglish(word) and len(word)>1 and (word not in stopW) and (lword not in stopW) and (stword not in stopW)):
                cleanContent.append(stword)
                # if it is a new word founded, add a new dimension to the vector space
                if (stword not in vectorBase):
                    vectorBase.append(stword)
                    vectorBaseCnt.append(1)
                # else, add the appearance count of the word
                elif (content.index(word) == i):
                    ind = vectorBase.index(stword)
                    vectorBaseCnt[ind] += 1
        
        # Get the merged contents after stopwords removed
        PR[1] = cleanContent
        
        # If the PR has words & all characters are legal
        if (len(cleanContent)>0) and judgeLegal(PR[4]):
            
            # Get the time of PR and process it into consistent format
            PR[5] = time.mktime(time.strptime(PR[5], "%Y-%m-%d %X"))
            PR[6] = time.mktime(time.strptime(PR[6], "%Y-%m-%d %X"))
            if (PR[5] < baseline):
                baseline = PR[5]
            if (PR[6] > deadline):
                deadline = PR[6]
            
            # Get authors involved in this PR. Add them into the author list of the project
            eList = PR[4].split(",")
            for eachAuthor in eList:
                newAuthor = Author(eachAuthor)
                authors.add(newAuthor)
                
            # Append processed PR into the PR lists
            PRs.append(PR)
    
    # Part A Score. Calculate tfidf for each PR and get its score in the vector space
    vectorScore = vectorSpace.tfidf(PRs, vectorBase, vectorBaseCnt, len(PRs))
    
    # Part C Score. Calculate common network scores among authors.
    baseline -= 24 * 3600
    relationScore = authors.makeRelations(PRs, baseline, deadline)

    # Check relations by getting cosine similarities of vector scores among PRs. Used for debugging
#    for i in range(len(PRs)-1):
 #       for j in range(i+1, len(PRs)):
  #          if expertise.cos(vectorScore[i], vectorScore[j])>0:
   #             print (expertise.cos(vectorScore[i], vectorScore[j]), PRs[i][4], PRs[j][4])
    
    return 0

# Running test dataset
def Test(file):
    global PRs, vectorBase, vectorBaseCnt, relationScore, vectorScore, authors, baseline, deadline
    
    # Maximum tolerance of differece as equalization. i.e. when abs(a-b)<minRel, we regard a=b
    minRel = 1e-10
    
    # Read test dataset
    csv_file = csv.reader(open(file,'r',errors='ignore'))
    
    # reset accuracy data
    predictCnt = 0
    correctCnt = 0
    actualCnt = 0
    
    # Set top-K estimation to top-5
    K = 5
    
    # For each line in test dataset
    for e, testcase in enumerate(csv_file):
        # if the testcase is not a PR, skip
        if (testcase[0] != "PR"):
            continue
        
        if (e > 100):
            break
        # Get title & content of the testcase. And merge them.
        contentTitle = word_tokenize(testcase[1])
        contentDetail = word_tokenize(testcase[2])
        contentTitle.extend(contentDetail)
        content = contentTitle
        
        # Remove stopwords
        cleanContent = []
        # iterate each word
        for i in range(len(content)):
            word = content[i]
            lword = word.lower()
            stword = p_stemmer.stem(lword)
            # if the word is a legal English word and not a stopword, add into the actual content list
            if (judgeEnglish(word) and len(word)>1 and (word not in stopW) and (lword not in stopW) and (stword not in stopW)):
                cleanContent.append(stword)
                
        # Get the merged contents after stopwords removed
        testcase[1] = cleanContent
        
        # If the PR has words & all characters are legal
        if (len(cleanContent)>0) and judgeLegal(testcase[4]):
            
            # Get vector score of the testcase in the vector space generated from the training dataset
            testScore = vectorSpace.tfidf([testcase], vectorBase, vectorBaseCnt, len(PRs))[0]
            
            # Find k closest PRs based on cosine similarities of vector scores, then calculate Expertise Scores for related authors; k = 5
            totalScore = [0.0]*authors.length()
            cosV = []
            index = []
            
            # cosine similarities with each PR in training dataset
            for i in range(len(PRs)):
                index.append(i)
                cosV.append(expertise.cos(vectorScore[i], testScore))
                
            # Pack scores & indices together. Sort them by cosine similarities. And get k largest ones.
            pac = list(zip(cosV, index))
            pac.sort()
            topK = pac[-K:]
         
            # For each PR in k largest similarity PRs. sc being the similarity score, ind being the index of PR in training dataset
            for sc, ind in topK:
                
                # if the score is equal to 0, there is no relation, which happens when only fewer than k PRs in training dataset are related to the testcase PR
                if sc == 0:
                    continue    
                
                # Get authors related to the training PR and add Expertise Scores for them
                usrList = PRs[ind][4].split(",")
                for eachUsr in usrList:
                    usrid = authors.find(eachUsr)[0]
                    totalScore[usrid] += sc
        
            # Add common network score, and check the result
            # dedic is the list of all authors related in this testcase PR
            dedic = testcase[4].split(",")
            
            # the one who submit the testcase PR
            contributor = dedic[0]
            
            # get his id in the author list generated from training data. 
            # If the contributor doesn't exist in the author list, existance will get a value of False, otherwise True
            [con_id, existance] = authors.find(contributor)
            
            index = []
            # Add common network scores for each author related to the contributor
            for i in range(authors.length()):
                index.append(i)
                # if the contributor exists in the author list from training dataset
                if existance:
                    totalScore[i] += relationScore[con_id][i]
                
            # Pack total scores & indices together. Sort them by total scores. And get k largest ones.
            pac = list(zip(totalScore, index))
            pac.sort()
            topKusr = pac[-K:]
            
            # K closest authors to the testcase PR are predicted. Add K into the predict counter. For Precision
            predictCnt += len(topKusr)
            # The list of k closest authors to the PR
            predictList = []
            # sc being the total score of the author, ind being its index in author list
            for sc, ind in topKusr:
                name = authors.getName(ind)
                predictList.append(name)
                # if the author predicted is in the testcase result, it is correctly predicted
                if name in dedic:
                    correctCnt += 1
            
            # Add all authors related to this testcase PR into counters for Recall
            actualCnt += len(set(dedic))
            # Print the predict list and actual list of authors related to the testcase PR
            # print(predictList, set(dedic), authors.find(contributor)[1])
            # print (correctCnt, predictCnt, actualCnt)
            # print(" ")
    
    # Final Precision & Recall of the project
    print ("Precision:", float(correctCnt)/predictCnt, "Recall:", float(correctCnt)/actualCnt)
            


if __name__ == "__main__":
    
    # Adjust maxsize to successfully load large .csv files
    maxInt = sys.maxsize
    decrement = True
    while decrement:
        # decrease the maxInt value by factor 2
        # as long as the OverflowError occurs.
        decrement = False
        try:
            csv.field_size_limit(maxInt)
        except OverflowError:
            maxInt = int(maxInt/2)
            decrement = True
    
    # Get all project folders in the list
    reviews = []
    file_dir = "./archive/"
    for root, dirs, files in os.walk(file_dir):
        for d in dirs:
            reviews.append(os.path.join(root, d))
    
    # process each project
    for each in reviews:
        print (each)
        Train(each+"/training_data.csv")
        Test(each+"/testing_data.csv")
        # break