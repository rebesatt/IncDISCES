import os
from matplotlib import pyplot as plt
from math import ceil
import seaborn as sns
import pandas as pd
import random
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def main():
    gen_plot_eqsupp_l1()
    # gen_plot_difsupp_l()

def gen_plot_difsupp_l():
    n = random.randint(10,10000)
    
    xs = []
    ys = []
    zs = []
    color_list = []
    count = 0
    repeat_number = 10000
    
    for _ in range(repeat_number):

        supp0= 1- random.random()
        supp1 = 1- random.random()

        if supp0> supp1:
            temp_supp = supp0
            supp0 = supp1
            supp1 = temp_supp
        
        
        l = random.randint(1,n)

        s0 = ceil(supp0*n)
        sl = ceil(supp1*(n+l))
        if sl >= s0+l:
            color = 'tab:blue'
            count+=1
        else:
            color = 'tab:orange'
        xs.append(supp0)
        ys.append(supp1)
        zs.append(l)
        color_list.append(color)

    print(count/repeat_number)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(zs, xs, ys, c=color_list)
    plt.show()




def gen_plot_eqsupp_l1():
    x= [10, 50, 100, 500, 1000]
    n= max(x)
    
    # df = pd.DataFrame(columns= range(n+1), index=x)
    # m_list = [[] for _ in range(n+1)]
    # for j in x:
    #     for m in range(1,j+1):
    #         low_support = (m-1)/(j)
    #         high_support = m/(j+1)
    #         support1 = high_support - low_support
    #         support2 = m/j -  high_support
    #         m_list[m].append(support1)
        
    #         df.loc[j, 2*m] = support1
    #         df.loc[j, 2*m+1] = support2

    # plt.rcParams.update({'font.size': 22})
    # df.to_csv('support_values1.csv')
    # df.plot.bar(stacked=True, legend=False, color=sns.color_palette("tab10", n_colors=2), figsize=(10, 10))
    
    # plt.legend(labels=["$s_1<s_0+1$", "$s_1=s_1+0$"], mode = "expand", ncol = 2)
    # plt.xticks(rotation=0)
    # plt.xlabel('number of streams')
    # plt.ylabel('support')
    # # plt.title('Support distribution for different number of streams')
    # plt.savefig('img/support_distribution1.pdf')

    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True)
    
    for counter, k in enumerate([1,2,5]):
        file_path = f'support_values_{k}.csv'
        df = pd.DataFrame(columns= range(n+k), index=x)
            
        m_list = [[] for _ in range(n+k)]
        for j in x:
            for m in range(1,j+1):
                if m > j*(k-1)/k:
                    low_support = (m-1)/(j)
                    high_support = (m-1+k)/(j+k)
                    support1 = max(0,high_support - low_support)
                    support2 = max(0,m/j -  high_support)
                    m_list[m].append(support1)
                else:
                    support1 = m/j - (m-1)/(j)
                    support2 = 0
                df.loc[j, 2*m] = support1
                df.loc[j, 2*m+1] = support2

        
        # df.to_csv(file_path)
        df.plot.bar(stacked=True, legend=False, 
                    color=sns.color_palette("colorblind", n_colors=2), 
                    figsize = (8,9),
                    fontsize = 14, rot =0 , ax=axes[counter])
            
        
        # axes[counter].set_xticks( rotation=0)
        
        # axes[counter].set_ylabel('support', fontsize =20)
        axes[counter].set_title(f'k={k}', fontsize = 17)
        # plt.title('Support distribution for different number of streams')
    plt.rcParams.update({'font.size': 18})
    fig.suptitle('Support distribution for different number of streams')
    fig.supylabel('support')
    fig.supxlabel('number of streams')
    
    fig.tight_layout()
    fig.set_size_inches(9,5)
    plt.legend(labels=["$s_1<s_0+k$", "$s_1=s_0+k$"],
                loc='lower right')
            #    bbox_to_anchor=(0.5, 1.2), loc='upper left',
            #    ncols= 2)
    fig.savefig(f'experiments/results/support_distribution_all.pdf')
    


if __name__ == "__main__":
    main()