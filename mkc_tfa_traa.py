import os
import time
import numpy as np
from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI

#import torch
#import torchvision.datasets
#import torchvision.transforms as transforms

from scipy.optimize import linear_sum_assignment as lsa
from more_itertools import set_partitions
import random
import copy
from prettytable import PrettyTable





for run_agent in range(2000,2001,1):
#for run_agent in range(20,21,1):
    n_agents=run_agent
    n_tasks= 50
    agent=n_agents
    task=n_tasks
    noc=n_tasks
    features=5
    
    solution_table = PrettyTable(['Data Set', 'MKCS', 'MKGA', 'MKHC', 'MKSA'])
    time_table = PrettyTable(['Data Set', 'MKCS', 'MKGA','MKHC', 'MKSA'])
    
    myfile=open(str(run_agent) + " Agents.txt", "w")
    ####################### DATA ##############################
    
    for run_data in range(6):
        if run_data==0:
            ##### UPD #######
            agents=np.random.uniform(0, 1, (n_agents,features))
            tasks=np.random.uniform(0, 1, (n_tasks,features))
            agent_data = agents
            task_data = tasks
        elif run_data==1:
            # ##### NPD #######
            agents = np.random.normal(1, 0.01, (n_agents, features))
            tasks = np.random.normal(1, 0.01, (n_tasks, features))
            agent_data = agents
            task_data = tasks
        elif run_data==2:
            # ##### SUPD #####
            agents=np.random.uniform(0, 0.1, (n_agents,features))   
            tasks=np.random.uniform(0, 0.1, (n_tasks,features)) 
            agent_data = agents
            task_data = tasks
        elif run_data==3:
            # ##### SNPD #####   
            agents=np.random.normal(0.1, 0.01, (n_agents,features)) 
            tasks=np.random.normal(0.1, 0.01, (n_tasks,features)) 
            agent_data = agents
            task_data = tasks
        elif run_data==4:
            ########## F Distributions ##############
            agents=np.random.f(1,n_agents,size=(n_agents,features))
            tasks=np.random.f(1,n_tasks,size=(n_tasks,features))
            agent_data = agents
            task_data = tasks
        elif run_data==5:
            ######## Beta ###########
            agents=( np.random.beta(0.5, 0.5,size=(n_agents,features)) )
            tasks=( np.random.beta(0.5, 0.5,size=(n_tasks,features))  )
            agent_data = agents
            task_data = tasks
        
        def mkcsga(K, max_iter=300, n_runs=5):
            n_kernels = K.shape[0]
            # Run multiple times as results depend on initial P
            max_cost = +np.inf
            for v_runs in range(n_runs):
                # Randomly initialize P
                P = np.random.rand(n_kernels, K.shape[1], centers.shape[0])
                for j in range(n_kernels):
                    P[j], _ = np.linalg.qr(P[j])
                
                # Main optimization loop
                cost = +np.inf
                for i in range(max_iter):
                    # Update S
                    B = np.zeros((P.shape[1], P.shape[2]))
                    for j in range(n_kernels):
                        B += (K[j] @ (K[j].T @ P[j]))
                    S = np.clip(B, 0, 1)
        
                    # Update P
                    for j in range(n_kernels):
                        U, _, Vt = np.linalg.svd(K[j] @ (K[j].T @ S), full_matrices=False)
                        P[j] = U @ Vt
        
                    # Compute cost
                    prev_cost = cost
                    Z = np.zeros((P.shape[2], P.shape[1]))
                    for j in range(n_kernels):
                        Z += P[j].T @ K[j] @ K[j].T
                    cost = -np.trace(Z @ S)
                    if np.abs(prev_cost - cost) < 1e-9:
                        break
        
                # Track best results
                if max_cost > cost:
                    max_cost = cost
                    final_S = np.array(S)
                    final_P = np.array(P)
        
            # Form weights based on KL-divergence of reconstructed kernels with an ideal cluster similarity matrix
            w = np.zeros((n_kernels,))
            re_K = np.empty(K.shape)
            for j in range(n_kernels):
                # reconstruct kernels
                re_K[j] = S @ (P[j].T @ K[j])
                # min-max standardize
                max_idx = re_K[j].argmax(axis=1)
                re_K[j] = ((re_K[j] - re_K[j].min(axis=1)[:,None] + 1e-9) 
                    / (re_K[j].max(axis=1)[:,None] - re_K[j].min(axis=1)[:,None] + 2e-9))
                # form ideal cluster similarity matrix
                base = np.zeros(re_K[j].shape)
                base[np.arange(base.shape[0]), max_idx] = 1
                # KL-divergence
                #w[j] = (- re_K[j] * np.log(base / re_K[j])).sum(axis=1).mean()
                # Cross-Entropy
                w[j] = -(base * np.log(re_K[j]) +  (1 - base) * np.log(1 - re_K[j])).mean(axis=1).mean()
            # Form weights
            #w = (w / w.sum()) ** 0.5
            w = np.exp(-w)
            w = w / w.sum()
            #print(w)
            #print(sum(w))
            # Combine reconstructed kernels
            comb_K = (w[:,None,None] * re_K).sum(axis=0)
            return comb_K
        
        kernel_start = time.time()
        if __name__ == '__main__':    
                n_clusters = noc
                X = agents
                centers=tasks
                # centers = np.zeros((n_clusters, X.shape[1]))
                # for j in range(n_clusters):
                #     centers[j] = X[y==j].mean(axis=0)
        
                # Form kernel matrices
                n_kernels = 12
                K = np.empty((n_kernels, X.shape[0], centers.shape[0]))
                sqdist = cdist(X, centers)
                K[0] = np.exp(-sqdist / (2 * sqdist.max() * (1e-2) ** 2))
                K[1] = np.exp(-sqdist / (2 * sqdist.max() * (5e-2) ** 2))
                K[2] = np.exp(-sqdist / (2 * sqdist.max() * (1e-1) ** 2))
                K[3] = np.exp(-sqdist / (2 * sqdist.max()))
                K[4] = np.exp(-sqdist / (2 * sqdist.max() * (10) ** 2))
                K[5] = np.exp(-sqdist / (2 * sqdist.max() * (50) ** 2))
                K[6] = np.exp(-sqdist / (2 * sqdist.max() * (100) ** 2))
                K[11] = X @ centers.T
                K[7] = ((1/2) * K[11]) ** 2
                K[8] = ((1/4) * K[11]) ** 4
                K[9] = ((1/2) * K[11] + 1) ** 2
                K[10] = ((1/4) * K[11] + 1) ** 4
                K[11] = K[11] / np.fmax(np.outer(
                    np.linalg.norm(X, axis=1, ord=2), np.linalg.norm(centers, axis=1, ord=2)
                ), 1e-9)
                # min-max standardize matrices
                for j in range(7,12):
                    K[j] = (K[j] - K[j].min()) / (K[j].max() - K[j].min())
                # orthogonalize matrices
                for j in range(12):
                    U, _, Vt = np.linalg.svd(K[j], full_matrices=False)
                    K[j] = U @ Vt
                    
                comb_K = mkcsga(K, max_iter=300, n_runs=5)
                #print('Combined K=\n', comb_K)
        
        kernel_end = time.time()
        kernel_time= (kernel_end - kernel_start)
        
        ################### Required Functions  ###########
        def solution_value(final_col_struct):    ## Final Solution Value
            all_coalition_value=[]
            for k in range (len(final_col_struct)):
                all_coalition_value.append(sum(comb_K[final_col_struct[k]][:,k]))
            return sum(all_coalition_value)  
        
        
        
        # LSA and TASKS SATISFY
        mmc_start=time.time()
        
        task_agent_dist = (comb_K).T
        
        unfulfilled_tasks = np.arange(n_tasks)
        free_agents = np.arange(n_agents)
        max_rounds = int(np.ceil(n_agents / n_tasks))
        assign_matrix = np.zeros((n_tasks, n_agents), dtype='int')
        
        
        for i in range(max_rounds):
            row_assign, col_assign = lsa(task_agent_dist[unfulfilled_tasks][:,free_agents], maximize=True)
            #print(i, unfulfilled_tasks[row_assign], free_agents[col_assign])
            assign_matrix[unfulfilled_tasks[row_assign], free_agents[col_assign]] = 1
            delete_tasks = []
            for tj in range(unfulfilled_tasks.shape[0]):
                task_idx = unfulfilled_tasks[tj]
                if ((tasks[task_idx] - agents[np.where(assign_matrix[task_idx])[0]].sum(axis=0)) > 0).sum() == 0:
                    delete_tasks.append(tj)
            unfulfilled_tasks = np.delete(unfulfilled_tasks, np.array(delete_tasks, dtype=int))
            free_agents = np.delete(free_agents, col_assign)
        
        #print(assign_matrix)
        assign_agents = np.zeros((n_agents,)) - 1
        for tj in range(n_tasks):
            assign_agents[np.where(assign_matrix[tj])[0]] = tj
        #print(assign_agents)
        
        """
        import matplotlib.pyplot as plt
        plt.figure()
        cols = ['r', 'g', 'b']
        for tj in range(n_tasks):
            plt.scatter(tasks[tj,0], tasks[tj,1], marker='o', c=cols[tj])
            plt.scatter(agents[assign_agents==tj][:,0], agents[assign_agents==tj][:,1], marker='x', c=cols[tj])
        plt.scatter(np.atleast_2d(agents[free_agents])[:,0], np.atleast_2d(agents[free_agents])[:,1], marker='d', c='y')
        """
        
        # Assign free agents
        while len(free_agents) > 0:
            row_assign, col_assign = lsa(task_agent_dist[:,free_agents], maximize=True)
            assign_matrix[row_assign, free_agents[col_assign]] = 1
            free_agents = np.delete(free_agents, col_assign)
        
        #print(assign_matrix)
        assign_agents = np.zeros((n_agents,)) - 1
        for tj in range(n_tasks):
            assign_agents[np.where(assign_matrix[tj])[0]] = tj
        #print(assign_agents)
        
        """
        import matplotlib.pyplot as plt
        plt.figure()
        cols = ['r', 'g', 'b']
        for tj in range(n_tasks):
            plt.scatter(tasks[tj,0], tasks[tj,1], marker='o', c=cols[tj])
            plt.scatter(agents[assign_agents==tj][:,0], agents[assign_agents==tj][:,1], marker='x', c=cols[tj])
        #plt.scatter(np.atleast_2d(agents[free_agents])[:,0], np.atleast_2d(agents[free_agents])[:,1], marker='d', c='y')
        plt.show()
        """
        
        # Final Coalition Structure Generation
        mmc_col_struct=[[] for i in range (n_tasks)]
        for i in range (n_tasks):
            for j in range (len(assign_matrix[i])):
                if assign_matrix[i][j]==1:
                    mmc_col_struct[i].append(j)
        
        mkcsga_sol_val=solution_value(mmc_col_struct)
        mmc_end=time.time()
        #print("MKCS Complete")
        # print("Solution Value of MMC=",solution_value(mmc_col_struct))
        # print("MMC Execution Time=",(mmc_end-mmc_start), "Seconds")
        # print("\nFinal Solution of MKCSGA=",mmc_col_struct)
        
        
        
        
              
               
        # ####################################
        greedy_start=time.time()
        ####### Greedy Approach
        all_agents=[i for i in range(n_agents)]
        random_agent_permutation=random.sample(all_agents, n_agents)
        
        greedy_coalition_struct=[[] for i in range(n_tasks)]
        for i in range(n_agents):
            temp_assign=[]
            for j in range(n_tasks):
                greedy_coalition_struct[j].append(random_agent_permutation[i])
                temp_assign.append( solution_value(greedy_coalition_struct) ) # finding total structure value
                #temp_assign.append( np.sum(dist_mat[greedy_coalition_struct[j],j]) ) # finding coalition value only
                greedy_coalition_struct[j].remove(random_agent_permutation[i])
            #print(temp_assign)
            greedy_coalition_struct[np.argmax(temp_assign)].append(random_agent_permutation[i])
        greedy_col_struct=greedy_coalition_struct  ## Greedy Solution
        greedy_primary_sol=solution_value(greedy_col_struct)
        # greedy_penalty=0
        # for k in range(len(greedy_col_struct)):
        #     if len(greedy_col_struct[k])==0:
        #         greedy_penalty +=1
        # if greedy_penalty > 0:
        #     greedy_penalty=greedy_primary_sol * ((greedy_penalty/n_tasks)*100)
        greedy_sol_val =greedy_primary_sol #+ greedy_penalty
        greedy_end=time.time()
        #print("Greedy Complete")
        
        
        
        # ########### Hill Climb Approach
        final_hill_time=[]
        final_hill_value=[]
        
           
        hill_start=time.time()
        # ##### Taking Random Coalition Structure 
        random_mem=np.random.randint(0,n_tasks,size=(1,n_agents))[0]
        random_col_struct=[[] for i in range(n_tasks)]
        for i in range(n_tasks):
            random_col_struct[i]=list(np.where(random_mem==i)[0])
        
        temp_lst=copy.deepcopy(random_col_struct)
        #print("Random=",temp_lst)
        primary_sol_val=solution_value(temp_lst)
        hill_values=[]
        hill_structs=[]
        
        for viter in range(10):
            #print("Iteration=",viter, "And Coalition Strcut=",temp_lst)
            for t in range(n_agents):
                temp_agent=random_agent_permutation[t]
                agent_idx=next(((i, temp_lst.index(temp_agent)) for i, temp_lst in enumerate(temp_lst) if temp_agent in temp_lst), None)[0]
                temp_lst[agent_idx].remove(temp_agent)
                temp_sol_val=[]
                for k in range(n_tasks):
                    temp_lst[k].append(temp_agent)
                    temp_sol_val.append(solution_value(temp_lst))
                    temp_lst[k].remove(temp_agent)
                #print(temp_sol_val)
                #print(primary_sol_val)
                
                if min(temp_sol_val) > primary_sol_val:
                    #print("YES")
                    temp_lst[np.argmin(temp_sol_val)].append(temp_agent)
                    primary_sol_val=min(temp_sol_val)
                    temp_lst=list(temp_lst)
                    #print(temp_lst)
                else:
                    #print("NO")
                    temp_lst[agent_idx].append(temp_agent)
                    temp_lst=list(temp_lst)
                    primary_sol_val=solution_value(temp_lst)
                    #print(temp_lst)
                    
            hill_structs.append(temp_lst)
            hill_values.append(primary_sol_val)
            #print("Hill =",temp_lst, "Value=",primary_sol_val)
        
        hill_col_struct=hill_structs[np.argmin(hill_values)] 
        hill_sol_val=solution_value(hill_col_struct)  
        hill_end=time.time()
        #print("Hill Climb Complete")
        
        
        
        #######################################################
        # ########### Simulated Annealing Approach #########################
        sim_start=time.time()
        # ##### Taking Random Coalition Structure 
        random_mem=np.random.randint(0,n_tasks,size=(1,n_agents))[0]
        random_col_struct=[[] for i in range(n_tasks)]
        for i in range(n_tasks):
            random_col_struct[i]=list(np.where(random_mem==i)[0])
        
        temp_lst=copy.deepcopy(random_col_struct)
        # print("Random=",temp_lst)
        primary_sol_val=solution_value(temp_lst)
        sim_values=[]
        sim_structs=[]
        
        
        for viter in range(10):
            #print("Iteration=",viter, "And Coalition Strcut=",temp_lst)
            for t in range(n_agents):
                rand_assign=np.random.randint(0,n_tasks, size=2)
                temp_agent=random_agent_permutation[t]
                agent_idx=next(((i, temp_lst.index(temp_agent)) for i, temp_lst in enumerate(temp_lst) if temp_agent in temp_lst), None)[0]
                temp_lst[agent_idx].remove(temp_agent)
                temp_sol_val=[]
                for k in range(2):
                    temp_lst[rand_assign[k]].append(temp_agent)
                    temp_sol_val.append(solution_value(temp_lst))
                    temp_lst[rand_assign[k]].remove(temp_agent)
                #print(temp_sol_val)
                #print(primary_sol_val)
                
                if min(temp_sol_val) > primary_sol_val:
                    #print("YES")
                    temp_lst[np.argmin(temp_sol_val)].append(temp_agent)
                    primary_sol_val=min(temp_sol_val)
                    temp_lst=list(temp_lst)
                    #print(temp_lst)
                else:
                    #print("NO")
                    temp_lst[agent_idx].append(temp_agent)
                    temp_lst=list(temp_lst)
                    primary_sol_val=solution_value(temp_lst)
                    #print(temp_lst)
                    
            sim_structs.append(temp_lst)
            sim_values.append(primary_sol_val)
            #print("Hill =",temp_lst, "Value=",primary_sol_val)
        
        sim_col_struct=sim_structs[np.argmin(sim_values)] 
        sim_primary_sol=solution_value(sim_col_struct)
        # sim_penalty=0
        # for k in range(len(sim_col_struct)):
        #     if len(sim_col_struct[k])==0:
        #         sim_penalty +=1
        # if sim_penalty > 0:
        #     sim_penalty=sim_primary_sol * ((sim_penalty/n_tasks)*100)
        sim_sol_val =sim_primary_sol #+ sim_penalty
        sim_end=time.time()
        #print("Sim Anneal Complete\n")
        #'''
        
        
        # print("Agent=",run_agent,file=myfile)
        # print("DATA=",run_data,file=myfile)
        # #################### Final Result Print
        # print("MKCS Value=", mkcsga_sol_val,file=myfile)
        # print("Greedy Value=", greedy_sol_val,file=myfile)
        # #print("Hill Value=", hill_sol_val,file=myfile)
        # print("Simulated Value=", sim_sol_val,file=myfile)
        # print(" \n")
        
        solution_table.add_row([run_data, mkcsga_sol_val, greedy_sol_val, hill_sol_val, 
                                       sim_sol_val])
        
        
        mkcsga_time= (mmc_end - mmc_start)
        greedy_time= (greedy_end - greedy_start)
        hill_time= (hill_end - hill_start)
        sim_time= (sim_end - sim_start)
        # print("MKCS Time=", mkcsga_time,file=myfile)
        # print("Greedy Time=", greedy_time,file=myfile)
        # #print("Hill Time=", hill_time,file=myfile)
        # print("Simulated Time=", sim_time,file=myfile)
        
        time_table.add_row([run_data, mkcsga_time, greedy_time, hill_time, 
                                 sim_time])
        
        #print("******************\n",file=myfile)
    
    
    print("Solution Values",file=myfile)
    print(solution_table,file=myfile)
    print("\nExecution Time",file=myfile)
    print(time_table,file=myfile)
    myfile.close()
    print(run_agent, "Agents With", n_tasks, "Tasks (Fixed) Completed\n")










