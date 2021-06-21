import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import datetime
data1="C:/Users/asus/Desktop/AUT/SI/t2/semeion/semeion_csv.csv"
data2="C:/Users/asus/Desktop/AUT/SI/t2/arrhythmia/arrhythmia_csv.csv"
data3="C:/Users/asus/Desktop/AUT/SI/t2/hillvalley/hill-valley_csv.csv"
data=data3

Data=pd.read_csv(data)
Data=Data.replace(np.nan, 0)
Data_x=Data.iloc[:,:-1]
Data_y=Data.iloc[:,-1]
feat_list=list(Data_x)

class Whale():
    def __init__(self,f_list=[]):
        self.f_list=f_list
        self.fitness=None
        
    def Fitness(self):
        Data=pd.read_csv(data)
        Data=Data.replace(np.nan, 0)
        Data_x=Data.iloc[:,:-1]
        Data_y=Data.iloc[:,-1]
        feat_list=list(Data_x)
        for i,f in enumerate(self.f_list):
            if f==0:Data_x=Data_x.drop([feat_list[i]],axis=1)
        knn=KNeighborsClassifier(n_neighbors=5)
        cv_scores=cross_val_score(knn,Data_x,Data_y,cv=10)
        error_rate=np.mean([1-acc for acc in cv_scores])
        self.f_list=[int(item) for item in self.f_list]
        F=self.f_list.count(1)
        self.fitness=(0.9*error_rate)+(0.1*F/len(self.f_list))
        
    def Acc(self):
        Data=pd.read_csv(data)
        Data=Data.replace(np.nan, 0)
        Data_x=Data.iloc[:,:-1]
        Data_y=Data.iloc[:,-1]
        feat_list=list(Data_x)
        for i,f in enumerate(self.f_list):
            if f==0:Data_x=Data_x.drop([feat_list[i]],axis=1)
        knn=KNeighborsClassifier(n_neighbors=5)
        cv_scores=cross_val_score(knn,Data_x,Data_y,cv=10)
        F=self.f_list.count(1)
        print (np.mean(cv_scores))
        print (F)

class pop():
    def __init__(self,res=None):
        self.res=res
        self.BestWhale=None
        
    def ordering_pop(self):
        for r in self.res:
            r.Fitness()
        ress=sort_by_fitness(self.res)
        self.BestWhale=ress[0]
        self.res=ress
        
def init_pop(n):
    init_list=[]
    for i in range(n):
        feat_list=np.zeros(len(list(Data_x)))
        feat_list=[np.random.choice([0,1]) for x in feat_list]
        init_list.append(Whale(f_list=feat_list))
    return init_list

def sort_by_fitness(lis):
    sorted_list=[]
    list1=[obj.fitness for obj in lis]
    while list1:
        minn=min(list1)
        for l in lis:
            if l.fitness==minn:
                sorted_list.append(l)
                lis.remove(l)
                list1=[obj.fitness for obj in lis]
                break
    return sorted_list

def Hamming_dis(lis1,lis2):
    if len(lis1) != len(lis2): return None
    dis=0
    for i in range(len(lis1)):
        if lis1[i]!=lis2[i]:dis+=1
    if dis!=0:return dis
    if dis==0:return 0.001

def WhaleAlgorithm(pop,epochs=50):
    def sig10(x):
        return 1/(1+np.exp(-10*(x)))
    def sig12(x):
        return 1/(1+np.exp(-1*(x)))
    def bstep(c):
        rand=np.random.uniform(0,1)
        if c>= rand : return 1
        if c<  rand : return 0
    Dim=len(pop.BestWhale.f_list)
    def yy(x,b):
        if (x+b) >=1 : return 1
        else : return 0
    for ep in range(epochs):
        print(f'percentage: {ep/epochs*100} %')
        pop.BestWhale.Acc()
        print(pop.BestWhale.fitness)
        print('')
        a=2-(2*ep/epochs)
        a=np.array([a for i in range(Dim)])
        r1=np.random.uniform(0,1)
        A=[((2*r1)-1)*i for i in a]
        for i in pop.res[1:]:
            D=np.zeros(Dim)
            C=np.random.uniform(0,2)
            c_step=np.zeros(Dim)
            b_step=np.zeros(Dim)
            X_new=np.zeros(Dim)
            roo=np.random.uniform(0,1)
            #h1=pop.BestWhale.f_list
            if roo <= 0.5:
                if A[0] >= 0:
                    for d in range(Dim):
                        D[d]=np.abs(C*pop.BestWhale.f_list[d]-i.f_list[d])
                        c_step[d]=sig10(A[d]*D[d]-0.5)
                        b_step[d]=bstep(c_step[d])
                        X_new[d]=yy(pop.BestWhale.f_list[d],b_step[d])
                if A[0] < 0:
                    rand=np.random.choice(range(len(pop.res)))
                    for d in range(Dim):
                        D[d]=np.abs(C*pop.res[rand].f_list[d]-i.f_list[d])
                        c_step[d]=sig10(A[d]*D[d]-0.5)
                        b_step[d]=bstep(c_step[d])
                        X_new[d]=yy(pop.BestWhale.f_list[d],b_step[d])
            if roo > 0.5:
                for d in range(Dim):
                    l=np.random.uniform(-1,1)
                    term=np.exp(l*a[0]/2)*np.cos(2*np.pi*l)
                    D[d]=np.abs(C*pop.BestWhale.f_list[d]-i.f_list[d])
                    c_step[d]=sig10(D[d]*term + pop.BestWhale.f_list[d])
                    b_step[d]=bstep(c_step[d])
                    X_new[d]=yy(pop.BestWhale.f_list[d],b_step[d])
            i.f_list=X_new
        pop.ordering_pop()
                

results=init_pop(20)
pop=pop(res=results)
pop.ordering_pop()
list1=[pop]   
a = datetime.datetime.now()         
WhaleAlgorithm(pop)
b = datetime.datetime.now()
c = b - a
print( int(c.total_seconds()))
list1=[pop]
