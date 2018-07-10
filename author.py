import math

# Function provided in the paper to calcualte time-related scores for common network
def calcTime(a, BaseL, DDL):
    return (a-BaseL)/(DDL-BaseL)

# class for each author
class Author(object):
    
    # author name
    __name = ""  
    
    # indices of PRs related to the author
    __PRList = []
    
    def __init__(self, nm):
        self.__name = nm
        self.__PRList = []
        
    def addPR(self, PRId):
        self.__PRList.append(PRId)
        return 0
        
    def __getPR(self):
        return self.__PRList
        
    # called when merge 2 author objects when 2 objects have the same author name. Literally only one author exists in fact
    def extendAuthor(self, otherAu):
        if self.__name != otherAu.getName():
            return 1
        self.__PRList.extend(otherAu.__getPR())        
        del(otherAu)
        return 0
        
    def getName(self):
        return self.__name
    
    # for test
    def getPR(self):
        return self.__PRList
     
# the author list of all authors in the training dataset
class AuthorList:
    
    # the list of all authors by ascending dictionary order of author names
    __l = []    
    
    # relation scores for each pair of authors
    __relations = []
    
    def __init__(self):
        self.__l = []
        
    def clear(self):
        self.__l = []
        
    ##############################################################################
    # find the author index in the list given the author name
    # because the list has an order, the search uses divide & conquer to save time
    #
    # Input:
    #   name: the author name
    #
    # Output:
    #   [ans, existance]
    #   ans: the index of author object in the author list. 
    #        If no such author is found by given input name, it will return the author with highest dictionary order which is lower than the given input name
    #   existance: boolean value of whether the given name exists in the author list
    ##############################################################################
    def find(self, name):
        head = 0
        tail = self.length() - 1
        while (head <= tail):
            mid = (head + tail)//2
            if self.__l[mid].getName() <= name:
                head = mid+1
            else:
                tail = mid-1
        ans = head - 1
        
        if (ans < 0):
            return [-1, False]
        return [ans, (self.__l[ans].getName() == name)]
    
    # return length of the author list
    def length(self):
        return len(self.__l)
    
    # add an author into the list, depending on whether it already exists in the list
    def add(self, au):
        name = au.getName()
        
        # try to find the author in the list
        ind = self.find(name)
        
        # if it does already exist
        if (ind[1]):
            self.__l[ind[0]].extendAuthor(au)
            return 1
        
        # if it doesn't exist, and it should be inserted into the list based on the search result
        elif (ind[0] < self.length()-1):    
            self.__l.insert(ind[0]+1, au)
            return 0
        # if it doesn't exist, and it should be appended to the end of the list based on the search result
        else:
            self.__l.append(au)
            return 0
    
    # calculate relations for each pair of authors, given all PRs in the dataset, the baseline and deadline time of all PRs
    def makeRelations(self, PRs, baseline, deadline):
        # author length
        alen = self.length()
        # initiate relation scores
        self.__relations = [[0.0 for i in range(alen)] for j in range(alen)]
        
        # two hyperparameters define in the paper
        relationConst = 1.0
        lam = 0.8
        
        # for each PR
        for PR in PRs:
            # [0]: PR   [1]: title   [2]: content    [4]: user_list   [5]: time1     [6]: time2
            
            # get involved authors to the PR
            usrList = []
            for eachUser in PR[4].split(","):
                ind = self.find(eachUser)
                # if the author doesn't exist in the list, there is error in training phase
                if (ind[1] == False):
                    print("ERROR",eachUser,ind[0])
                usrList.append(ind[0])
            
            # get numbers of all related authors to the PR
            L = len(usrList)
            
            # if only one author, it is the contributor, and as a result no common network is built
            if L < 2:
                continue
            
            # s_id: the contributor id
            s_id = usrList[0]
            
            # initialize numbers of appreance for each author in this PR
            cnt = [0] * alen
            
            # for each appearance of an author
            for i in range(L):
                # the author id
                t_id = usrList[i]
                
                # decay parameter by amount of appearance 
                decay = math.pow(lam, cnt[t_id])
                cnt[t_id] += 1
                
                # calculate common score from all parameters
                self.__relations[s_id][t_id] += decay * relationConst * calcTime(PR[6], baseline, deadline)
            
        return self.__relations
    
    def getName(self, index):
        return self.__l[index].getName()
       
    def display(self, leng = 0):
        if (leng == 0):
            leng = self.length()
        for i in range(leng):
            print (self.__l[i].getName())
    
                