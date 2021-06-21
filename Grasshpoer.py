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

class grasshoper():
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
        self.Target=None
        
    def ordering_pop(self):
        for r in self.res:
            r.Fitness()
        ress=sort_by_fitness(self.res)
        self.Target=ress[0]
        self.res=ress
        
def init_pop(n):
    init_list=[]
    for i in range(n):
        feat_list=np.zeros(len(list(Data_x)))
        feat_list=[np.random.choice([0,1]) for x in feat_list]
        init_list.append(grasshoper(f_list=feat_list))
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

def Grasshoper(pop,epochs=50):
    def S(r):
        return (0.5*np.exp(-r/1.5))-(np.exp(-r))
    def G(x):
        return ((2*x/(len(pop.Target.f_list)))+2)
    def defr(l1,l2):
        if len(l1)!=len(l2):return None
        l3=np.zeros(len(l1))
        for d in range(len(l1)):
            if l1[d]==l2[d]: l3[d]=0
            if l1[d]!=l2[d]: l3[d]=1
        return l3
    for ep in range(epochs):
        print(f'{ep/epochs*100} %')
        pop.Target.Acc()
        print(pop.Target.fitness)
        c=1-((ep/epochs)*(1-0.00001))
        for i in pop.res:
            if i==pop.Target:continue
            for j in pop.res:
                if j==i: continue
                Dist=np.zeros(len(i.f_list))
                f_Dist=np.zeros(len(i.f_list))
                deff=defr(i.f_list,j.f_list)
                dd=Hamming_dis(i.f_list,j.f_list)
                gg=G(dd)
                ss=S(gg)
                for d in range(len(pop.Target.f_list)):
                    Dist[d]+=(0.5*c*ss*deff[d]/np.sqrt(dd))
            Dist=[s*c for s in Dist]
            had=np.mean(Dist)
            #had=(max(Dist)-min(Dist))/2
            for d in range(len(pop.Target.f_list)):
                if Dist[d]>= had : f_Dist[d]=1
                if Dist[d]< had : f_Dist[d]=0
                else:
                    rand=np.random.uniform(0,1)
                    rand=np.round(rand,decimals=2)
                    if rand >= 0.5: f_Dist[d]=1
                    else: f_Dist[d]=0
                if f_Dist[d]==0: i.f_list[d]=pop.Target.f_list[d]
                if f_Dist[d]==1: i.f_list[d]=np.abs(pop.Target.f_list[d]-1)              
        pop.ordering_pop()
        

results=init_pop(20)
pop=pop(res=results)
pop.ordering_pop()
list1=[pop]
a = datetime.datetime.now()    
Grasshoper(pop)
b = datetime.datetime.now()
c = b - a
print( int(c.total_seconds()))
