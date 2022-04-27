# %%
# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from sklearn.tree import DecisionTreeRegressor


# %% [markdown]
# 1. 要查询配置对应的性能，也就是必须要有一个字典，键是配置，字符串；值是性能，float。
# 2. 已测量配置池，未测量配置池，都需要有增删的功能，因此，需要用集合来表示，且配置全部用字符串来表示
# 3. 模型中，配置却不能是字符串，而是数字，这样才能训练模型

# %%
class System:
    def __init__(self, loc="./Data/X264_AllMeasurements.csv", target_max=True, transform=None):
        data = pd.read_csv(loc).to_numpy().astype(np.float32)
        self.data = data
        
        X = data[:, :-1]
        y = data[:, -1:].flatten()
        if transform:
            y = transform(y)
        # self.row_y = data[:, -1:].flatten()
        self.conf_perf_dict = dict([(str(conf), perf) for (conf, perf) in zip(X, y)])
        
        data_tuple = [[conf, perf] for conf, perf in self.conf_perf_dict.items()]
        data_tuple = sorted(data_tuple, key=lambda x: x[1], reverse=target_max)
        # 最优、最差值
        self.best = data_tuple[0][1]
        self.worst = data_tuple[-1][1]
        # 方便查排名
        self.rank = dict([(conf, i+1) for (i, (conf, perf)) in enumerate(data_tuple)])
    
    def get_all_perfs(self):
        return np.array(list(self.conf_perf_dict.values()))
    
    def get_perf(self, X):
        """返回性能值
        Args:
            X (str): 配置字符串
        """
        return self.conf_perf_dict[X]
    
    def measure(self, conf):
        assert isinstance(conf, str)
        return self.get_perf(conf)
    
    def rank_difference(self, conf):
        """返回配置的排序差"""
        return self.rank[conf] - 1


# %%

def argmax_acquisition(model, uneval_configs, target_max=True):
    """对uneval_configs进行预测，返回最优配置"""
    X = np.array([np.array(list(map(float, key[1:-1].split()))) for key in uneval_configs])
    y = model.predict(X)
    if target_max:
        idx = np.argmax(y)
    else:
        idx = np.argmin(y)
    return str(X[idx])

def flash(loc, init_size, budget, target_max, save_dir, transform=None, remove=True, limit=None, draw=True):
    system = System(loc=loc, target_max=target_max, transform=transform)
    
    # 初始化未测量池为 全部配置，测量池为 空集
    uneval_configs = set(system.conf_perf_dict.keys())
    eval_configs = set()

    # 随机选取初始点
    random.seed(10)
    init_confs = random.sample(list(system.conf_perf_dict.keys()), init_size)
    
    # 测量初始点，并加入到测量集合中，从未测量集合中移除
    for conf in init_confs:
        if remove:
            uneval_configs.remove(str(conf))
        eval_configs.add(str(conf))

    # print(init_confs)
    # print(uneval_configs)
    
    rds = []
    preds = []
    best_rds = []
    best_preds = []
    best_rd = len(system.conf_perf_dict)
    best_pred = system.worst
    
    # 循环迭代建模
    for i in range(budget):
        # 建模
        X, y = [], []
        for conf in eval_configs:
            # str 转 ndarray
            X.append(np.array(list(map(float, conf[1:-1].split()))))
            y.append(system.get_perf(conf))
        X, y = np.array(X), np.array(y)
        dtr = DecisionTreeRegressor()
        dtr.fit(X, y)
        
        # 建模后，使用采样函数，从未测量集合中选取下一个测量的点
        acquired_conf = argmax_acquisition(dtr, uneval_configs, target_max=target_max)
        
        # 测量配置
        system.measure(acquired_conf)
        
        # 更新配置池
        eval_configs.add(acquired_conf)
        if remove:
            uneval_configs.remove(acquired_conf)
        
        # 查看性能如何
        # print('第{:>2}次选取配置测量后，其配置性能：{:.3f}，排名差：{}'.format(i+1, system.get_perf(acquired_conf), system.rank_difference(acquired_conf)))
        rds.append(system.rank_difference(acquired_conf))
        preds.append(system.get_perf(acquired_conf))
        
        best_rd = min(best_rd, system.rank_difference(acquired_conf))
        best_pred = max(best_pred, system.get_perf(acquired_conf)) if target_max else min(best_pred, system.get_perf(acquired_conf))
        best_rds.append(best_rd)
        best_preds.append(best_pred)
        # print(system.rank_difference(acquired_conf))

    if draw:
        # print(best_rds)
        # 绘图保存
        fig, axes=plt.subplots(1, 3, figsize=(18, 5))
        axes = axes.reshape(1, 3)
        name = loc.split("/")[-1][:-4]
        title = "Dataset: {}    Target: find {}    InitSize: {}".format(name, ("maxValue" if target_max else "minValue"), init_size)
        fig.suptitle(title)
        
        axes[0][0].set_title("all perf distribution")
        axes[0][0].set_ylabel("performance value")
        # axes[0][0].set_xticks([])
        all_perfs = system.get_all_perfs()
        axes[0][0].hist(all_perfs, 100, orientation=u'horizontal')
        # axes[0][0].scatter(range(len(all_perfs)), all_perfs, s=5)
        
        # 性能值图
        axes[0][1].set_title("predict_best - iterations")
        axes[0][1].plot(range(budget), [system.best] * budget, '--', label="real best perf")
        # if remove:
        axes[0][1].plot(range(budget), preds, label="pred current perf")
        axes[0][1].plot(range(budget), best_preds, label="pred best perf")
        # else:
        #     axes[0][1].plot(range(budget), preds, label="pred current perf")
        axes[0][1].set_ylabel("performance value")
        axes[0][1].set_xlabel("iterations ")
        axes[0][1].legend()
        
        # rank diff 图
        axes[0][2].set_title("rank_difference-iterations")
        axes[0][2].plot(range(budget), [0] * budget, "--", label="real best rank-diff")
        axes[0][2].plot(range(budget), rds, label="pred current rank-diff")
        axes[0][2].plot(range(budget), best_rds, label="pred best rank-diff")
        # if remove:
        #     axes[0][2].plot(range(budget), rds, label="pred current rank-diff")
        #     axes[0][2].plot(range(budget), best_rds, label="pred best rank-diff")
        # else:
        #     axes[0][2].plot(range(budget), rds, label="pred current rank-diff")
        #     axes[0][2].plot(range(budget), best_rds, label="pred best rank-diff")
        if limit:
            axes[0][2].set_ylim(0, limit)
            axes[0][2].set_ylabel("rank difference(limit {})".format(limit))
        else:
            axes[0][2].set_ylabel("rank difference")
        axes[0][2].set_xlabel("iterations ")
        axes[0][2].legend()
    
        if not os.path.exists("./results"):
            os.mkdir("./results")
        if not os.path.exists("./results/{}".format(save_dir)):
            os.mkdir("./results/{}".format(save_dir))
            
        fig.tight_layout()
        fig.savefig("./results/{}/{}.jpg".format(save_dir, name), bbox_inches='tight', transparent=False)
        # fig.savefig("./results/all_dataset_results/{}.jpg".format(name), bbox_inches='tight', transparent=False)
        plt.show()
        plt.close()
        
    # 返回 最好pred, 最好rd，最后pred，最后rd
    return best_pred, best_rd, preds[-1], rds[-1]

# %%
params = {
    'Apache_AllMeasurements.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": True,
        "note": "loads on the web server"
    },
    'BDBC_AllMeasurements.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": False,
        "note": "response time"
    }, 
    'WGet.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": False ,
        "note": "main memory"
    },
    'X264_AllMeasurements.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": False,
        "note": "encoding time"
    },
    'SQL_AllMeasurements.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": False,
        "note": "response time"
    },
    'lrzip.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": False,
        "note": "compressing time"
    },
    'Dune.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": False,
        "note": "偏微分方程求解工具，暂定性能是求解时间"
    },
    'HSMGP_num.csv': {
        "init_size": 100,
        "budget": 50,
        "target_max": False,
        "note": "Average Time Per Iteration"
    },
    'rs-6d-c3_obj1.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": True,
        "note": "吞吐量 throughput"
    }, 
    'rs-6d-c3_obj2.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": False,
        "note": "延迟 latency"
    },
    'sol-6d-c2-obj1.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": True,
        "note": "throughput"
    },
    'sol-6d-c2-obj2.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": False,
        "note": "latency"
    },
    'sort_256_obj2.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": True,
        "note": "FPGA sort 256 throughput"
    },
    'wc+rs-3d-c4-obj1.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": True,
        "note": "throughput"
    },
    'wc+rs-3d-c4-obj2.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": False,
        "note": "latency"
    },
    'wc+sol-3d-c4-obj1.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": True,
        "note": "throughput"
    },
    'wc+sol-3d-c4-obj2.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": False,
        "note": "latency"
    },
    'wc+wc-3d-c4-obj1.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": True,
        "note": "throughput"
    },
    'wc+wc-3d-c4-obj2.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": False,
        "note": "latency"
    },
    'wc-3d-c4_obj2.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": False,
        "note": "latency"
    },
    'wc-6d-c1-obj1.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": True,
        "note": "throughput"
    },
    'wc-6d-c1-obj2.csv': {
        "init_size": 20,
        "budget": 50,
        "target_max": False,
        "note": "latency"
    }
}


# %% [markdown]
# # 1 论文里的描述
# 测过的配置之后就不会再测了，记录测到的最好的值

# %%
def run(params, remove, save_dir):
    files = os.listdir("./Data")

    best_preds = []
    best_rds = []
    last_preds = []
    last_rds = []


    # # 对每个数据集运行FLASH，并保存对应结果和汇总结果
    # save_dir = "remove_evaled_configs"
    if os.path.exists("./results/{}/pred.txt".format(save_dir)):
        os.remove("./results/{}/pred.txt".format(save_dir))
    for file in files:
        print("flash {}...".format(file))
        best_pred, best_rd, last_pred, last_rd = flash(
            loc="./Data/{}".format(file),
            init_size=params[file]["init_size"],
            budget=params[file]["budget"],
            target_max=params[file]["target_max"],
            save_dir=save_dir,
            remove=remove
        )
        # best_pred, best_rd, last_pred, last_rd = flash("./Data/{}".format(file), init_size=init_size, budget=budget, target_max=is_target_max[file], remove=remove)
        best_preds.append(best_pred)
        best_rds.append(best_rd)
        last_preds.append(last_pred)
        last_rds.append(last_rd)
        with open("./results/{}/pred.txt".format(save_dir), "a+") as f:
            f.write(file + "\n")
            f.write("{} {} {} {}\n".format(best_pred, best_rd, last_pred, last_rd))

    # 历史最好rank-diff图
    # plt.scatter(range(len(files)), best_rds, label="best rd in iterations")
    # plt.xticks(range(len(files)), files, rotation=90)
    # plt.title("best rank difference")
    # plt.tight_layout()
    # # plt.savefig("./results/{}/best_rank_difference.jpg".format(save_dir))
    # plt.cla()
    
    # # 最后一次迭代的rank-diff图
    # plt.scatter(range(len(files)), last_rds)
    # plt.xticks(range(len(files)), files, rotation=90)
    
    plt.scatter(range(len(files)), best_rds, label="best rd in iterations", marker='^')
    plt.scatter(range(len(files)), last_rds, label="last rd in iterations")
    plt.xticks(range(len(files)), files, rotation=90)
    plt.legend()
    plt.title("rank difference")
    plt.tight_layout()
    plt.savefig("./results/{}/rank_difference.jpg".format(save_dir))


# %%
run(params, True, "remove_eval_configs")

# %%
import copy
new_params = copy.deepcopy(params)
for key in params.keys():
    new_params[key]['init_size'] = params[key]["init_size"] * 2
    new_params[key]['budget'] = int(params[key]["budget"] * 1.5)


# %%
run(new_params, False, "reserve_eval_configs")

# %%
for file in os.listdir("./Data"):
    system = System(loc="./Data/{}".format(file))
    print("{:25}options_num: {:<8}target: {:15}note: {}".format(file[:-4], system.data.shape[1]-1, "max perf" if params[file]["target_max"] else "min perf", params[file]["note"]))

# %%



