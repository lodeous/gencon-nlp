import numpy as np
from matplotlib import pyplot as plt
from lifelines.statistics import logrank_test
import pandas as pd

data = pd.read_csv('total_summary.csv', header=0, index_col=0)

def Kaplan_Meier(T, E):
    """T : times of events
       E : how many events are associated to the corresponding entry in T"""
    #start with n0, that is everyone
    n = len(T)
    n0 = n
    ni = n0
    steps = []
    varss = []
    end = max(T)
    #print(end)
    S_hat = []
    Var_hat = []
    for i in range(end):
        #get rid of people who have already left the study
        #the people whoes times have passed
        Ti = np.array(T[T>=i])
        Ei = np.array(E[T>=i])
        #we want the people that fulfill the below criteria
        mask = [True if (Ti[j] != i or Ei[j] != 0) else False for j in range(len(Ti))]
        #censored_mask = np.logical_or(Ti, Ei, where=Ti != i and  Ei != 0)
        Tf = Ti[mask]
        Ef = Ei[mask]
        if i == 0:
            ni = n
        else:
            ni = len(Tf)
        #get the people who "died"
        event_happened = Tf[Ef == 1]
        #get the people who "died" at time t
        event_happened_i = event_happened[event_happened == i]
        di = len(event_happened_i)
        #calculate step for time i
        frac = (ni-di)/ni
        steps.append(frac)
        S_hat_t = np.prod(steps)
        S_hat.append(S_hat_t)
        #Use Greenwood's Formula
        var = di/(ni*(ni-di))
        varss.append(var)
        #with a sum so everything isn't zero
        Var_hat_t = (S_hat_t**2)*np.sum(varss)
        Var_hat.append(Var_hat_t)
    return S_hat, Var_hat

S, V = Kaplan_Meier(data['Time'], data['Event'])



N = max(data['Time'])
domain = np.linspace(1, N, N)
plt.plot(domain, S, c='green', drawstyle='steps-post', label='KM Estimate')
y1 = S + np.sqrt(V)
y2 = S - np.sqrt(V)
plt.fill_between(domain, y1, y2, step='post', color='lightgreen')
plt.xlabel('Number of Words said before \'Christ\' is said in the Talk')
plt.ylabel('Estimated Probability of Not Saying Christ')
plt.title('Kaplan Meier Survival Function Estimating\nthe Probability of Saying Christ in General Conference')
plt.legend(loc='best')
plt.savefig('event_time_total.png')
plt.show()

#apostle mask
apostle = data['Apostle'] == True
data_apostle = data[apostle]
no_apostle = data[~apostle]

S_apostle, V_apostle = Kaplan_Meier(data_apostle['Time'], data_apostle['Event'])
S_no, V_no = Kaplan_Meier(no_apostle['Time'], no_apostle['Event'])

N_yes = max(data_apostle['Time'])
N_no = max(no_apostle['Time'])
domain_apostle = np.linspace(1, N_yes, N_yes)
domain_no = np.linspace(1, N_no, N_no)
plt.plot(domain_apostle, S_apostle, c='green', drawstyle='steps-post', label='Apostles')
y1_yes = S_apostle + np.sqrt(V_apostle)
y2_yes = S_apostle - np.sqrt(V_apostle)
plt.fill_between(domain_apostle, y1_yes, y2_yes, step='post', color='lightgreen')

plt.plot(domain_no, S_no, c='blue', drawstyle='steps-post', label='Other Speakers')
y1_no = S_no + np.sqrt(V_no)
y2_no = S_no - np.sqrt(V_no)
plt.fill_between(domain_no, y1_no, y2_no, step='post', color='lightblue')

plt.xlabel('Number of Words said before \'Christ\' is said in the Talk')
plt.ylabel('Estimated Probability of Not Saying Christ')
plt.title('Kaplan Meier Survival Function Estimating Probability\nof Saying Christ in General Conference Talk')
plt.legend(loc='best')
plt.savefig('event_time_apostle.png')
