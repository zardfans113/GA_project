# 答案 y = 11
import numpy as np
# import math


# ==== 參數設定(與問題相關) ====

NUM_JOB = 20            # 工件個數   # === Step 1. 設定成 3 個工件 ===
NUM_MACHINE = 15                      # === Step 1. 設定 3 個機台

# === Step 2. 設定 processing time 矩陣(row: job ID; col: 對應其加工機台所用的加工時間) ===
pTime = [[24, 12, 17, 27, 21, 25, 27, 26, 30, 31, 18, 16, 39, 19, 26],[ 30, 15, 20, 19, 24, 15, 28, 36, 26, 15, 11, 23, 20, 26, 28],[ 35, 22, 23, 32, 20, 12, 19, 23, 17, 14, 16, 29, 16, 22, 22],[ 20, 29, 19, 14, 33, 30, 32, 21, 29, 24, 25, 29, 13, 20, 18],[ 23, 20, 28, 32, 16, 18, 24, 23, 24, 34, 24, 24, 28, 15, 18],[ 24, 19, 21, 33, 34, 35, 40, 36, 23, 26, 15, 28, 38, 13, 25],[ 27, 30, 21, 19, 12, 27, 39, 13, 12, 36, 21, 17, 29, 17, 33],[ 27, 19, 29, 20, 21, 40, 14, 39, 39, 27, 36, 12, 37, 22, 13],[ 32, 29, 24, 27, 40, 21, 26, 27, 27, 16, 21, 13, 28, 28, 32],[ 35, 11, 39, 18, 23, 34, 24, 11, 30, 31, 15, 15, 28, 26, 33],[ 28, 37, 29, 31, 25, 13, 14, 20, 27, 25, 31, 14, 25, 39, 36],[ 22, 25, 28, 35, 31, 21, 20, 19, 29, 32, 18, 18, 11, 17, 15],[ 39, 32, 36, 14, 28, 37, 38, 20, 19, 12, 22, 36, 15, 32, 16],[ 28, 29, 40, 23, 34, 33, 27, 17, 20, 28, 21, 21, 20, 33, 27],[ 21, 34, 30, 38, 11, 16, 14, 14, 34, 33, 23, 40, 12, 23, 27],[ 13, 40, 36, 17, 13, 33, 25, 24, 23, 36, 29, 18, 13, 33, 13],[ 25, 15, 28, 40, 39, 31, 35, 31, 36, 12, 33, 19, 16, 27, 21],[ 22, 14, 12, 20, 12, 18, 17, 39, 31, 31, 32, 20, 29, 13, 26],[ 18, 30, 38, 22, 15, 20, 16, 17, 12, 13, 40, 17, 30, 38, 13],[ 31, 39, 27, 14, 33, 31, 22, 36, 16, 11, 14, 29, 28, 22, 17]]

# === Step 2. 設定 job 的機台加工順序 ===
mOrder = [
    [2, 3, 9, 4, 0, 6, 8, 7, 1, 5, 11, 14, 13, 10, 12],
[6, 3, 12, 11, 1, 13, 10, 2, 5, 7, 0, 8, 14, 9, 4],
[6, 0, 13, 7, 2, 3, 12, 10, 9, 1, 5, 11, 8, 4, 14],
[9, 6, 1, 7, 12, 4, 0, 5, 11, 10, 14, 2, 3, 8, 13],
[11, 13, 1, 6,7,5,8,9,3,10,2,0,14,12,4],
[8, 11, 14, 1, 7, 6, 5, 10, 3, 2, 4, 9, 13, 12, 0],
[13, 3, 6, 8, 12, 4, 2, 9, 14, 5, 10, 11, 1, 0, 7],
[5, 4, 6, 9, 3, 10, 8, 14, 13, 2, 1, 12, 11, 7, 0],
[13, 11, 8, 3, 5, 4, 9, 0, 14, 6, 2, 10, 7, 12, 1],
[12, 1, 5, 14, 7, 0, 3, 13, 8, 11, 4, 10, 2, 9, 6],
[10, 5, 12, 1, 7, 8, 14, 4, 3, 9, 13, 11, 6, 2, 0],
[0, 11, 5, 13, 4, 8, 9, 14, 2, 7, 10, 1, 3, 12, 6],
[12, 5, 2, 8, 3, 13, 0, 6, 7, 11, 14, 1, 4, 9, 10],
[8, 1, 14, 12, 4, 5, 6, 10, 0, 7, 11, 2, 13, 9, 3],
[9, 14, 3, 12, 0, 11, 2, 5, 1, 8, 4, 13, 10, 6, 7],
[9, 14, 7, 4, 0, 5, 8, 13, 10, 3, 2, 1, 11, 6, 12],
[3, 5, 2, 12, 7, 1, 8, 6, 11, 4, 10, 14, 9, 13, 0],
[12, 10, 0, 2, 5, 1, 11, 8, 14, 3, 7, 9, 13, 4, 6],
[5, 10, 7, 14, 13, 11, 9, 3, 1, 2, 12, 6, 8, 4, 0],
[9, 8, 12, 1, 5, 3, 11, 13, 0, 7, 14, 4, 6, 2, 10],
]


# ==== 參數設定(與演算法相關) ====

NUM_ITERATION = 100	# 世代數(迴圈數)

NUM_CHROME = 40		# 染色體個數
NUM_BIT = NUM_JOB * NUM_MACHINE		   # 染色體長度 # === Step 3-1. 編碼是 000111222 的排列 ===

Pc = 50  					# 交配率 (代表共執行Pc*NUM_CHROME/2次交配)
Pm = 0.01   					# 突變率 (代表共要執行Pm*NUM_CHROME*NUM_BIT次突變)

NUM_PARENT = NUM_CHROME                         # 父母的個數
NUM_CROSSOVER = int(Pc * NUM_CHROME / 2)        # 交配的次數
NUM_CROSSOVER_2 = NUM_CROSSOVER*2               # 上數的兩倍
NUM_MUTATION = int(Pm * NUM_CHROME * NUM_BIT)   # 突變的次數# === Step 3-2. NUM_BIT 要修改成 3 x 3 ===

np.random.seed(0)          # 若要每次跑得都不一樣的結果，就把這行註解掉

# ==== 基因演算法會用到的函式 ====    # === Step 5. 設定適應度函數 ===
def initPop():             # 初始化群體
    p = []

    # === 編碼 000111222 的排列  ===
    for i in range(NUM_CHROME) :        
        a = []
        for j in range(NUM_JOB):
            for k in range(NUM_MACHINE):
                a.append(j)
        np.random.shuffle(a)

        p.append(a)
        
    return p

def fitFunc(x):            # 適應度函數
    S = np.zeros((NUM_JOB, NUM_MACHINE))    # S[i][j] = Starting time of job i at machine j
    C = np.zeros((NUM_JOB, NUM_MACHINE))    # C[i][j] = Completion time of job i at machine j
    
    B = np.zeros(NUM_MACHINE, dtype=int)    # B[j] = Available time of machine j
    
    opJob = np.zeros(NUM_JOB, dtype=int)    # opJob[i] = current operation ID of job i
    
    for i in range(NUM_BIT):
        m = mOrder[x[i]][opJob[x[i]]]
        if opJob[x[i]] != 0:
            S[x[i]][m] = max([B[m], C[x[i]][mOrder[x[i]][opJob[x[i]]-1]]])
        else:
            S[x[i]][m] = B[m]
            
        C[x[i]][m] = B[m] = S[x[i]][m] + pTime[x[i]][opJob[x[i]]]
        
        opJob[x[i]] += 1
            
    return -max(B)           # 因為是最小化問題

def evaluatePop(p):        # 評估群體之適應度
    return [fitFunc(p[i]) for i in range(len(p))]

def selection(p, p_fit):   # 用二元競爭式選擇法來挑父母
	a = []

	for i in range(NUM_PARENT):
		[j, k] = np.random.choice(NUM_CHROME, 2, replace=False)  # 任選兩個index
		if p_fit[j] > p_fit[k] :                      # 擇優
			a.append(p[j].copy())
		else:
			a.append(p[k].copy())

	return a

def crossover_one_point(p):           # 用單點交配來繁衍子代 (new)
	a = []

	for i in range(NUM_CROSSOVER) :
		c = np.random.randint(1, NUM_BIT)      		  # 隨機找出單點(不包含0)
		[j, k] = np.random.choice(NUM_PARENT, 2, replace=False)  # 任選兩個index
     
		child1, child2 = p[j].copy(), p[k].copy()
		remain1, remain2 = list(p[j].copy()), list(p[k].copy())     # 存還沒被用掉的城市
       
		for m in range(NUM_BIT):
			if m < c :
				remain2.remove(child1[m])   # 砍掉 remain2 中的值是 child1[m]
				remain1.remove(child2[m])   # 砍掉 remain1 中的值是 child2[m]
		
		t = 0
		for m in range(NUM_BIT):
			if m >= c :
				child1[m] = remain2[t]
				child2[m] = remain1[t]
				t += 1
		
		a.append(child1)
		a.append(child2)

	return a

def crossover_uniform(p):           # 用均勻交配來繁衍子代 (new)
	a = []

	for i in range(NUM_CROSSOVER) :
		mask = np.random.randint(2, size=NUM_BIT)
		[j, k] = np.random.choice(NUM_PARENT, 2, replace=False)  # 任選兩個index
     
		child1, child2 = p[j].copy(), p[k].copy()
		remain1, remain2 = list(p[j].copy()), list(p[k].copy())     # 存還沒被用掉的城市
       
		for m in range(NUM_BIT):
			if mask[m] == 1 :
				remain2.remove(child1[m])   # 砍掉 remain2 中的值是 child1[m]
				remain1.remove(child2[m])   # 砍掉 remain1 中的值是 child2[m]
		
		t = 0
		for m in range(NUM_BIT):
			if mask[m] == 0 :
				child1[m] = remain2[t]
				child2[m] = remain1[t]
				t += 1
		
		a.append(child1)
		a.append(child2)

	return a
def mutation(p):	           # 突變
	for _ in range(NUM_MUTATION) :
		row = np.random.randint(NUM_CROSSOVER_2)  # 任選一個染色體
		[j, k] = np.random.choice(NUM_BIT, 2, replace=False)  # 任選兩個基因
      
		p[row][j], p[row][k] = p[row][k], p[row][j]       # 此染色體的兩基因互換


def sortChrome(a, a_fit):	    # a的根據a_fit由大排到小
    a_index = range(len(a))                         # 產生 0, 1, 2, ..., |a|-1 的 list
    
    a_fit, a_index = zip(*sorted(zip(a_fit,a_index), reverse=True)) # a_index 根據 a_fit 的大小由大到小連動的排序
   
    return [a[i] for i in a_index], a_fit           # 根據 a_index 的次序來回傳 a，並把對應的 fit 回傳

def replace(p, p_fit, a, a_fit):            # 適者生存
    b = np.concatenate((p,a), axis=0)               # 把本代 p 和子代 a 合併成 b
    b_fit = p_fit + a_fit                           # 把上述兩代的 fitness 合併成 b_fit
    
    b, b_fit = sortChrome(b, b_fit)                 # b 和 b_fit 連動的排序
    
    return b[:NUM_CHROME], list(b_fit[:NUM_CHROME]) # 回傳 NUM_CHROME 個為新的一個世代


# ==== 主程式 ====

pop = initPop()             # 初始化 pop
pop_fit = evaluatePop(pop)  # 算 pop 的 fit

best_outputs = []                           # 用此變數來紀錄每一個迴圈的最佳解 (new)
best_outputs.append(np.max(pop_fit))        # 存下初始群體的最佳解

mean_outputs = []                           # 用此變數來紀錄每一個迴圈的平均解 (new)
mean_outputs.append(np.average(pop_fit))        # 存下初始群體的最佳解

for i in range(NUM_ITERATION) :
    parent = selection(pop, pop_fit)            # 挑父母
    offspring = crossover_one_point(parent)     # 單點交配
    mutation(offspring)                         # 突變
    offspring_fit = evaluatePop(offspring)      # 算子代的 fit
    pop, pop_fit = replace(pop, pop_fit, offspring, offspring_fit)    # 取代

    best_outputs.append(np.max(pop_fit))        # 存下這次的最佳解
    mean_outputs.append(np.average(pop_fit))    # 存下這次的平均解

    print('iteration %d: x = %s, y = %d'	%(i, pop[0], -pop_fit[0]))     # fit 改負的

# 畫圖
import matplotlib.pyplot
matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.plot(mean_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()