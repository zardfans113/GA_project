# 找 x 使得最大化 f(x) = 1024 - x^2
# x 用 6 個 binary bit 編碼

import numpy as np
# import math

# ==== 參數設定(與演算法相關) ====

NUM_ITERATION = 20			# 世代數(迴圈數)

NUM_CHROME = 20				# 染色體個數 (一定要偶數)
NUM_BIT = 9					# 染色體長度

Pc = 0.5    					# 交配率 (代表共執行Pc*NUM_CHROME/2次交配)
Pm = 0.01   					# 突變率 (代表共要執行Pm*NUM_CHROME*NUM_BIT次突變)
penalty_cost = 1000             # 惩罚成本
NUM_PARENT = NUM_CHROME                         # 父母的個數
NUM_CROSSOVER = int(Pc * NUM_CHROME / 2)        # 交配的次數
NUM_CROSSOVER_2 = NUM_CROSSOVER*2               # 上數的兩倍
NUM_MUTATION = int(Pm * NUM_CHROME * NUM_BIT)   # 突變的次數

#np.random.seed(0)          # 若要每次跑得都不一樣的結果，就把這行註解掉

# ==== 基因演算法會用到的函式 ====
def initPop():             # 初始化群體
	return np.random.randint(2, size=(NUM_CHROME,NUM_BIT)) # 產生 NUM_CHROME 個二元編碼

def fitFunc(x):            # 適應度函數
    p_count = 0
	
    if x[0] + x[1] + x[2] > 1:
        p_count += 1
    if x[3] + x[4] + x[5] > 1:
         p_count += 1
    if x[6] + x[7] + x[8] > 1:
        p_count += 1
    if x[0] + x[3] + x[6] > 1:
        p_count += 1
    if x[1] + x[4] + x[7] > 1:
        p_count += 1
    if x[2] + x[5] + x[8] > 1:
        p_count += 1
    if x[0] + x[4] + x[8] > 1:
        p_count += 1
    if x[2] + x[4] + x[6] > 1:
        p_count += 1
    denominator = 10 * x[0] + 9 * x[1] + 13 * x[2] + 14 * x[3] + 12 * x[4] + 11 * x[5] + 12 * x[6] + 15 * x[7] + 16 * x[8] + penalty_cost * p_count
    if denominator == 0:
        return 0
    else:
        return 1/denominator

def evaluatePop(p):        # 評估群體之適應度
    return [fitFunc(p[i]) for i in range(len(p))]

def selection(p, p_fit):   # 用二元競爭式選擇法來挑父母
	a = []

	for i in range(NUM_PARENT):
		[j, k] = np.random.choice(NUM_CHROME, 2, replace=False)  # 任選兩個index
		if p_fit[j] > p_fit[k] :                      # 擇優
			a.append(p[j])
		else:
			a.append(p[k])

	return a

def crossover(p):           # 用單點交配來繁衍子代
	a = []

	for i in range(NUM_CROSSOVER) :
		c = np.random.randint(1, NUM_BIT)      		  # 隨機找出單點(不包含0)
		[j, k] = np.random.choice(NUM_PARENT, 2, replace=False)  # 任選兩個index
       
		a.append(np.concatenate((p[j][0: c], p[k][c: NUM_BIT]), axis=0))
		a.append(np.concatenate((p[k][0: c], p[j][c: NUM_BIT]), axis=0))

	return a

def mutation(p):	           # 突變
	for _ in range(NUM_MUTATION) :
		row = np.random.randint(NUM_CROSSOVER_2)  # 任選一個染色體
		col = np.random.randint(NUM_BIT)          # 任選一個基因
      
		p[row][col] = (p[row][col] + 1) % 2       # 對應此染色體的此基因01互換
        

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
best_outputs.append(np.max(pop_fit))        # 存下初始群體的最佳解 (new)

mean_outputs = []                           # 用此變數來紀錄每一個迴圈的平均解 (new)
mean_outputs.append(np.average(pop_fit))        # 存下初始群體的最佳解 (new)

for i in range(NUM_ITERATION) :
    parent = selection(pop, pop_fit)            # 挑父母
    offspring = crossover(parent)               # 交配
    mutation(offspring)                         # 突變
    offspring_fit = evaluatePop(offspring)      # 算子代的 fit
    pop, pop_fit = replace(pop, pop_fit, offspring, offspring_fit)    # 取代
    
    best_outputs.append(np.max(pop_fit))        # 存下這次的最佳解 (new)
    mean_outputs.append(np.average(pop_fit))    # 存下這次的平均解 (new)

    print('iteration %d: x = %s, y = %f'	%(i, pop[0], pop_fit[0]))
    

# 畫圖 (new)
import matplotlib.pyplot
matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.plot(mean_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()