import pulp as lp
import json
import sys
import os

#假设有20个

def pulp_slove(xtr):
    all_ctcvr = xtr["all_ctcvr"]
    credit_ctcvr = xtr["credit_ctcvr"]
    credit_activate = xtr["credit_activate"]
    #求解变量，是否要选取本次为曝光机会
    X = [lp.LpVariable(f"x_{i}", 0, 1, lp.LpInteger) for i in range(len(all_ctcvr))]
    #定义最小化问题
    prob = lp.LpProblem("myProblem", lp.LpMaximize)

    #优化目标
    prob += lp.lpSum([lp.lpDot(credit_activate[i], X[i]) for i in range(len(X))])
    #约束
    prob += (lp.lpSum([lp.lpDot(all_ctcvr[i] - credit_ctcvr[i], X[i]) for i in range(len(X))]) <= 1)
    prob += (lp.lpSum([lp.lpDot(1, X[i]) for i in range(len(X))]) <= 10)
    #求解
    status = prob.solve()

    #组合结果输出
    result = {
        "idx": [],
        "opt": 0.0,
        "constraint": 0.0
    }
    for i in range(len(X)):
        if X[i].value() == 1:
            result["idx"].append("%s" % i)
            result["opt"] += credit_activate[i]
            result["constraint"] += (all_ctcvr[i] - credit_ctcvr[i])
    return result


def my_slove(xtr, alpha = 0.0001, max_loop=200):
    #自己手撸一个........
    all_ctcvr = xtr["all_ctcvr"]
    credit_ctcvr = xtr["credit_ctcvr"]
    credit_activate = xtr["credit_activate"]
    
    #初始化lambda
    lambda_ = 0.0
    X = []
    for loop in range(max_loop):
        #计算x_i，只要大于0就保留
        X = []
        if loop < 10:
            alpha_ = 0.1
        elif loop < 50:
            alpha_ = 0.01
        elif loop < 100:
            alpha_ = 0.001
        else:
            alpha_ = alpha
        
        #这段比较坑，如果lambda太大，会导致x_i都为0，导致梯度为0，无法更新lambda
        """
        for x in range(len(credit_activate)):
            if credit_activate[x] - lambda_ * (all_ctcvr[x] - credit_ctcvr[x]) > 0:
                X.append(1)
            else:
                X.append(0)
                
        """
        #修改后的逻辑，按松弛后值排序，从大到小选择，直到约束超了
        score_list = []
        X=[0 for _ in credit_activate]

        for i in range(len(credit_activate)):
            score = credit_activate[i] - lambda_ * (all_ctcvr[i] - credit_ctcvr[i])
            score_list.append((i, score))

        score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
        constraint = 0.0

        for item in score_list:
            i = item[0]
            if (all_ctcvr[i] - credit_ctcvr[i]) + constraint > 1:
                continue
            else:
                X[i] = 1
                constraint += (all_ctcvr[i] - credit_ctcvr[i])

        #计算lambda，最小化约束条件
        grad =  (sum([(all_ctcvr[i] - credit_ctcvr[i]) * X[i] for i in range(len(X))]) - 1)
        lambda_ = lambda_ + alpha_ * grad
        lambda_ = max(lambda_, 0)

    result = {
        "idx": [],
        "opt": 0.0,
        "constraint": 0.0,
        "lambda": lambda_
    }

    for i in range(len(X)):
        if X[i] == 1:
            result["idx"].append("%s" % i)
            result["opt"] += credit_activate[i]
            result["constraint"] += (all_ctcvr[i] - credit_ctcvr[i])
    return result

def main():
    with open(sys.argv[1], "r") as reader:
        records = json.load(reader)

    debug_idx = []
    
    with open(sys.argv[2], "w") as writer:
        for idx, record in enumerate(records):
            if len(debug_idx) > 0 and idx not in debug_idx:
                continue
            
            if len(debug_idx) > 0:
                writer.write("[D][%s]\t" % idx + json.dumps(record) + "\n")
            result = pulp_slove(record)
            writer.write("[P][%s]\t" % idx + json.dumps(result) + "\n")
            result = my_slove(record)
            writer.write("[M][%s]\t" % idx + json.dumps(result) + "\n")
            writer.flush()

if __name__ == "__main__":
    main()