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

class Salp():
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
        self.FoodSource=None
        self.Leader=None
        
    def ordering_pop(self):
        for r in self.res:
            r.Fitness()
        ress=sort_by_fitness(self.res)
        self.FoodSource=ress[0]
        self.Leader=ress[1]
        self.res=ress
        
def init_pop(n):
    init_list=[]
    for i in range(n):
        feat_list=np.zeros(len(list(Data_x)))
        feat_list=[np.random.choice([0,1]) for x in feat_list]
        init_list.append(Salp(f_list=feat_list))
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

def SALP(pop,epochs=50):
    for ep in range(epochs):
        print(f'{ep/epochs*100} %')
        pop.FoodSource.Acc()
        print(pop.FoodSource.fitness)
        print('')
        next_leader=Salp(f_list=np.zeros(len(list(Data_x))))
        next_res=[]
        for i in range(len(next_leader.f_list)):
            omega=0.7
            r1=2*np.exp(-1*(-4*ep/epochs)**2)
            r2=np.random.uniform(0,1)
            r3=np.random.uniform(0,1)
            if r3>= 0.5: next_leader.f_list[i]=(omega*pop.FoodSource.f_list[i])+(r1*r2)
            if r3<  0.5: next_leader.f_list[i]=(omega*pop.FoodSource.f_list[i])-(r1*r2)
            next_leader.f_list[i]-=((omega)/2)
            next_leader.f_list[i]=1/(1+np.exp(-10*next_leader.f_list[i]))
            rand=np.random.uniform(0,1)
            if next_leader.f_list[i] >= rand : next_leader.f_list[i]=0
            else : next_leader.f_list[i]=1
        for m in range(len(pop.res[2:])):
            n_res=np.zeros(len(pop.res[m+2].f_list))
            for i in range(len(pop.res[m+2].f_list)):
                n_res[i]=0.5*(pop.res[m+2].f_list[i]+(omega*pop.res[m+1].f_list[i]))
                n_res[i]-=((1+omega)/4)
                n_res[i]=1/(1+np.exp(-10*n_res[i]))
                rand=np.random.uniform(0,1)
                if n_res[i] >= rand : n_res[i]=0
                else : n_res[i]=1   
            nn_res=[int(item) for item in n_res]
            next_res.append(Salp(f_list=nn_res))
        next_res.append(next_leader)
        next_res.append(pop.FoodSource)
        pop.res=next_res
        pop.ordering_pop()
        
        
results=init_pop(20)
pop=pop(res=results)
pop.ordering_pop()
list1=[pop]
a = datetime.datetime.now()           
SALP(pop)
b = datetime.datetime.now()
c = b - a
print( int(c.total_seconds()))
list1=[pop]
            
            

    