# -*- coding: utf-8 -*-
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC

parameters_ov_dict = dict(
    params_dtc={#已解决
        'estimator__random_state': [1],
        'estimator__criterion': ['gini', 'entropy'],#不纯度的计算方法，填写gini使用基尼系数，填写entropy使用信息增益
        'estimator__max_depth': np.arange(5, 20, 1),#限制树的最大深度
        'estimator__splitter': ['random', 'best'],#用来控制决策树中的随机选项的，有两种输入值，输入”best"，决策树在分枝时虽然随机，但是还是会优先选择更重要的特征进行分枝（重要性可以通过属性feature_importances_查看），输入“random"，决策树在分枝时会更加随机。'random'避免过拟合。
        'estimator__min_samples_split': np.arange(5, 30, 5),#min_samples_split限定，一个节点必须要包含至少min_samples_split个训练样本，这个节点才允许被分枝
        'estimator__min_samples_leaf': np.arange(1,30,5),#min_samples_leaf限定，一个节点在分枝后的每个子节点都必须包含至少min_samples_leaf个训练样本
        'estimator__min_impurity_decrease': [*np.linspace(0,0.5,20)],#min_impurity_decrease限制信息增益的大小，信息增益小于设定数值的分枝不会发生
        'estimator__max_features': [np.arange(5, 30, 5),"auto", "log2", "sqrt"]#max_features限制分枝时考虑的特征个数，防止过拟合，数字或者"auto", "sqrt", "log2"
        #'estimator__class_weight': ['balanced',None],#用于样本不平衡的数据，与min_weight_fraction_leaf搭配使用
        #'estimator__min_weight_fraction_leaf': [0.1,0.2,0.3,0.4,0.5]#与min_samples_leaf类似，但考虑了各个分类的权重,min_weight_fraction_leaf must in [0, 0.5]
    },
    params_rf={#已解决
        'estimator__random_state': [1],
        # 'estimator__max_depth': np.arange(3, 20, 5),
        'estimator__n_estimators': [1300],
        # 'estimator__max_leaf_nodes': np.arange(1, 50, 10),
        'estimator__criterion': ['entropy'],
        # 'estimator__min_samples_split': np.arange(2, 30, 5),
        # 'estimator__min_samples_leaf': np.arange(2, 30, 5),
        # 'estimator__max_features': np.arange(2, 35, 5),
        # 'estimator__min_impurity_decrease': [*np.linspace(0, 0.5, 10)],
        # 'estimator__class_weight': ['balanced',None],#用于样本不平衡的数据，与min_weight_fraction_leaf搭配使用
        # 'estimator__min_weight_fraction_leaf': [0.1,0.2,0.3,0.4,0.5]#与min_samples_leaf类似，但考虑了各个分类的权重,min_weight_fraction_leaf must in [0, 0.5]
        'estimator__n_jobs': [-1],
        'estimator__bootstrap': [True],#'bootstrap': [True, False],#默认True, Whether bootstrap samples are used when building trees.
        'estimator__oob_score': [True],#当n和n_estimators都不够大的时候，很可能就没有数据掉落在袋外，自然也就无法使用oob数据来测试模型了
        'estimator__max_samples': [0.9]
    },
    params_xgbc={#已解决，https://blog.csdn.net/qq_41076797/article/details/102710299
        'estimator__booster':['gbtree'],#gbtree 树模型做为基分类器
        # 'estimator__learning_rate': [0.1, 0.2],#学习率，控制每次迭代更新权重时的步长，默认0.3,典型值为0.01-0.2
        # 'estimator__early_stopping_rounds': [100,150],#在验证集上，当连续n次迭代，分数没有提高后，提前终止训练。
        'estimator__n_estimators': [500],#总共迭代的次数，即决策树的个数
        # 'estimator__max_depth': [2,10],#树的深度，默认值为6，典型值3-10。和GBM中的参数相同，这个值为树的最大深度。
        # 'estimator__min_child_weight': [1, 2],#值越大，越容易欠拟合；值越小，越容易过拟合（值较大时，避免模型学习到局部的特殊样本）
        'estimator__seed': [1],
        # 'estimator__subsample': [0.8, 0.6],#训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1
        # 'estimator__colsample_bytree': [0.8, 0.6],#训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1
        # 'estimator__gamma': [0, 0.1, 0.2, 0.5],#惩罚项系数，指定节点分裂所需的最小损失函数下降值。在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。 这个参数的值越大，算法越保守。
        # 'estimator__reg_alpha': [0.1,0.5,1],#L1正则化系数，默认为1
        # 'estimator__reg_lambda': [0.1,0.5,1],#L2正则化系数，默认为1
        # 'estimator__silent': [0],#时，不输出中间过程（默认）,=1输出中间过程
        # 'estimator__objective': ['binary'],#'multi:softprob'   num_class=n  返回概率,softmax返回类别;binary:logistic概率 ,binary:logitraw类别
        # 'estimator__num_class': [0],#类别,二分类时应该注释掉或者写为0
        # #'estimator__scale_pos_weight': [10],#当正负样本比例失衡时，设置正样本的权重，模型效果更好。例如，当正负样本比例为1:10时，scale_pos_weight=10
        'estimator__nthread': [-1]#时，使用全部CPU进行并行运算（默认）
    },
    params_adaboost={
        'estimator__base_estimator': [DTC(max_depth=8)],
        'estimator__n_estimators': [550],
        'estimator__learning_rate': [0.18],
        # 'estimator__algorithm': ['SAMME', 'SAMME.R'],
        # 'estimator__base_estimator__criterion': ['gini', 'entropy'],
        'estimator__base_estimator__splitter': ['best', 'random'],
        # 'estimator__base_estimator__max_depth': [7, 10, 15],
        # 'estimator__base_estimator__min_samples_split': [2,10,15],
        # 'estimator__base_estimator__min_samples_leaf': [10, 20],#如果达不到这个阈值，则同一父节点的所有叶子节点均被剪枝
        # 'estimator__base_estimator__max_features': [np.arange(5, 30, 5), 'auto', 'log2', 'sqrt', None],
        # 'estimator__base_estimator__max_leaf_nodes': [np.arange(5, 30, 5), None],
        # 'estimator__base_estimator__min_impurity_split': [1e-7, 0, None],
        'estimator__random_state': [1]
    },
    params_bagging={
        'estimator__base_estimator': [DTC()],#默认DTC
        'estimator__n_estimators': [50, 100],
        'estimator__bootstrap': [True, False]
    },
    params_gdbc={#已解决
        # 'estimator__learning_rate': [0.05, 0.1, 0.2],
        'estimator__n_estimators': [500],
        # 'estimator__subsample': [0.7, 0.8, 0.9, 1],
        # 'estimator__loss': ["deviance", "exponential"],
        # 'estimator__min_samples_split': [np.arange(5, 50, 5)],
        # 'estimator__min_samples_leaf': [np.arange(5, 50, 5)],
        # 'estimator__max_depth': [np.arange(5, 15, 3)],
        # 'estimator__max_features': [np.arange(15, 30, 5), None],
        # 'estimator__min_impurity_split': [1e-7, 0],
        'estimator__random_state': [1]
    },
    params_lightgbm={#https://www.biaodianfu.com/lightgbm.html #https://blog.csdn.net/qq_41076797/article/details/103225319?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166047473116782388067276%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=166047473116782388067276&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-14-103225319-null-null.nonecase&utm_term=sklearn&spm=1018.2226.3001.4450
        'estimator__random_state': [1],
        'estimator__boosting_type':['gbdt'],#提升树的类型 gbdt,dart,goss,rf
        # 'estimator__num_leavel':[32],#树的最大叶子数，需小于2^(max_depth)
        # 'estimator__max_depth': [-1,5],#-1表示最大深度
        # 'estimator__early_stopping_rounds': [100,150],
        # 'estimator__learning_rate': [0.1, 0.05],
        'estimator__n_estimators': [500],
        # 'estimator__objective': ['binary'],#binary,multiclass
        # 'estimator__num_class': [0],
        # 'estimator__min_child_weight': [0.01, 0.001],#分支结点的最小权重
        # 'estimator__subsample': [0.8, 0.9, 1],
        # 'estimator__colsample_bytree': [0.8],#训练特征采样率 列
        # 'estimator__reg_alpha': [0,0.5,1],
        # 'estimator__reg_lambda': [0,0.5,1],
        # 'estimator__n_jobs': [-1],
        # 'estimator__min_split_gain': [0],#最小分割增益
        # 'estimator__min_child_samples': [20, 30]
    },
    params_knn={#已解决
        'estimator__weights': ['uniform', 'distance'],
        'estimator__n_neighbors': [i for i in range(1, 5)],
        'estimator__p': [i for i in range(1, 3)],#控制Minkowski度量方法的值,整型，默认为2
        'estimator__n_jobs': [-1],
        'estimator__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']#计算n近邻的算法
    },
    params_catboost={
        'estimator__depth': [8],#树深，最大16，建议在1到10之间。默认6
        'estimator__learning_rate': [0.1],#学习率默认0.03
        # 'estimator__l2_leaf_reg': [1, 2, 3, 4],#正则参数。默认3
        'estimator__iterations': [700],#iterations 最大树数。默认1000
        # 'estimator__sampling_frequency': ['PerTree', 'PerTreeLevel'],#设置创建树时的采样频率，可选值PerTree/PerTreeLevel，默认为PerTreeLevel
        # 'estimator__random_leaf_estimation_method': ['Newton', 'Gradient'],#计算叶子值的方法，Newton/ Gradient。默认Gradient。
        'estimator__random_seed': [1],
        # # 'estimator__class_weights': [None],#类别的权重。默认None。
        # # 'estimator__class_scale_pos_weight': [None],  #二进制分类中class 1的权重。该值用作class 1中对象权重的乘数
        # 'estimator__approx_on_full_history': [False, True],# 计算近似值，False：使用1／fold_len_multiplier计算；True：使用fold中前面所有行计算。默认False。
        # 'estimator__random_strength': [0.2, 0.5, 1],#分数标准差乘数。默认1。
        # 'estimator__bootstrap_type': ['Bayesian', 'Bernoulli', 'No'],#定义权重计算逻辑，默认为Bayesian
        # 'estimator__bagging_temperature': [0.2,0.5,1]#贝叶斯套袋控制强度，区间[0, 1]。默认1
    }
)


parameters_dict = dict(
    params_dtc={#已解决
        'random_state': [1],
        'criterion': ['gini', 'entropy'],#不纯度的计算方法，填写gini使用基尼系数，填写entropy使用信息增益
        'max_depth': np.arange(5, 30, 1),#限制树的最大深度
        'splitter': ['random', 'best'],#用来控制决策树中的随机选项的，有两种输入值，输入”best"，决策树在分枝时虽然随机，但是还是会优先选择更重要的特征进行分枝（重要性可以通过属性feature_importances_查看），输入“random"，决策树在分枝时会更加随机。'random'避免过拟合。
        'min_samples_split': np.arange(5, 30, 5),#min_samples_split限定，一个节点必须要包含至少min_samples_split个训练样本，这个节点才允许被分枝
        'min_samples_leaf': np.arange(1,50,5),#min_samples_leaf限定，一个节点在分枝后的每个子节点都必须包含至少min_samples_leaf个训练样本
        'min_impurity_decrease': [*np.linspace(0,0.5,20)],#min_impurity_decrease限制信息增益的大小，信息增益小于设定数值的分枝不会发生
        'max_features': [np.arange(5, 30, 5),"auto", "log2", "sqrt"]#max_features限制分枝时考虑的特征个数，防止过拟合，数字或者"auto", "sqrt", "log2"
        #'class_weight': ['balanced',None],#用于样本不平衡的数据，与min_weight_fraction_leaf搭配使用
        #'min_weight_fraction_leaf': [0.1,0.2,0.3,0.4,0.5]#与min_samples_leaf类似，但考虑了各个分类的权重,min_weight_fraction_leaf must in [0, 0.5]
    },
    params_rf={#已解决
        #调参：https://www.bbsmax.com/A/o75NRBN5W3/
        'random_state': [1],
        'n_jobs': [-1],

        'n_estimators': [1200],
        'bootstrap': [True],
        # 'bootstrap': [True, False],#默认True, Whether bootstrap samples are used when building trees.
        'oob_score': [True],  # 当n和n_estimators都不够大的时候，很可能就没有数据掉落在袋外，自然也就无法使用oob数据来测试模型了
        'max_samples': [0.9],

        # 'max_depth': [np.arange(5, 20, 1), None],
        # 'max_leaf_nodes': [np.arange(5, 30, 5), None],
        'criterion': ['gini'],
        # 'min_samples_split': [np.arange(5, 30, 5)],
        # 'min_samples_leaf': [np.arange(5, 30, 5)],
        # 'max_features': [np.arange(10, 30, 5), "auto", "log2", "sqrt"],
        # 'min_impurity_decrease': [np.linspace(0, 0.5, 20), None],
        # 'class_weight': ['balanced', None],#用于样本不平衡的数据，与min_weight_fraction_leaf搭配使用
        # 'min_weight_fraction_leaf': [0.1, 0.2, 0.3, 0.4, 0.5, None]#与min_samples_leaf类似，但考虑了各个分类的权重,min_weight_fraction_leaf must in [0, 0.5]

    },
    params_xgbc={#已解决，https://blog.csdn.net/qq_41076797/article/details/102710299
        'seed': [1],
        'nthread': [-1],  # 时，使用全部CPU进行并行运算（默认）

        'booster':['gbtree'],#gbtree 树模型做为基分类器
        # 'learning_rate': [0.05, 0.1, 0.15, 0.2],#学习率，控制每次迭代更新权重时的步长，默认0.3,典型值为0.01-0.2
        # 'early_stopping_rounds': [100],#在验证集上，当连续n次迭代，分数没有提高后，提前终止训练。
        'n_estimators': [300],#总共迭代的次数，即决策树的个数np.arange(80, 300, 20)
        # 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],  # 训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1
        #
        #
        # 'max_depth': [np.arange(3, 20, 1), None], #树的深度，默认值为6，典型值3-10。和GBM中的参数相同，这个值为树的最大深度。
        # 'min_child_weight': [1, 2],#值越大，越容易欠拟合；值越小，越容易过拟合（值较大时，避免模型学习到局部的特殊样本）
        # 'colsample_bytree': [0.8, 0.6],#训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1
        # 'gamma': [0, 0.1, 0.2, 0.5],#惩罚项系数，指定节点分裂所需的最小损失函数下降值。在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。 这个参数的值越大，算法越保守。
        # 'reg_alpha': [0.1, 0.5, 1],#L1正则化系数，默认为1
        # 'reg_lambda': [0.1, 0.5, 1],#L2正则化系数，默认为1
        # 'silent': [0],#时，不输出中间过程（默认）,=1输出中间过程
        # 'objective': ['binary:logistic'],#'multi:softprob'   num_class=n  返回概率,softmax返回类别;binary:logistic概率 ,binary:logitraw类别
        # 'num_class': [0]#类别,二分类时应该注释掉或者写为0
        #'scale_pos_weight': [10],#当正负样本比例失衡时，设置正样本的权重，模型效果更好。例如，当正负样本比例为1:10时，scale_pos_weight=10

    },
    params_adaboost={
        'random_state': [1],
        # 'n_jobs': [-1],# 报错

        'base_estimator': [DTC(max_depth=7)],
        'n_estimators': [1100],
        'learning_rate': [0.1],
        # 'algorithm': ['SAMME', 'SAMME.R'],
        #
        # 'base_estimator__criterion': ['gini', 'entropy'],
        # 'base_estimator__splitter': ['best', 'random'],
        # 'base_estimator__max_depth': [7],
        # 'base_estimator__min_samples_split': [2, 10, 15],
        # 'base_estimator__min_samples_leaf': [10, 20],#如果达不到这个阈值，则同一父节点的所有叶子节点均被剪枝
        # 'base_estimator__max_features': [np.arange(5, 30, 5), 'auto', 'log2', 'sqrt', None],
        # 'base_estimator__max_leaf_nodes': [np.arange(5, 30, 5), None],
        # 'base_estimator__min_impurity_split': [1e-7, 0, None]

    },
    params_bagging={
        'base_estimator': [DTC()],#默认DTC
        'n_estimators': [50, 100],
        'bootstrap': [True, False]
    },
    params_gdbc={#已解决
        # 调参：https://www.bbsmax.com/A/LPdo6KwG53/
        'random_state': [1],
        # 'n_jobs': [-1] # 报错

        # 'learning_rate': [0.05, 0.1, 0.15, 0.2],
        'n_estimators': [500],
        # 'subsample': [0.8, 0.9, 1],
        # 'loss': ["deviance", "exponential"],
        #
        # 'min_samples_split': [np.arange(5, 30, 5), None],
        # 'min_samples_leaf': [np.arange(5, 30, 5), None],
        # 'max_depth': np.arange(3, 20, 1),
        # 'max_features': [np.arange(10, 30, 5), None],
        # 'max_leaf_nodes': [0],
        # 'min_impurity_split': [1e-7, 0]

    },
    params_lightgbm={#https://www.biaodianfu.com/lightgbm.html #https://blog.csdn.net/qq_41076797/article/details/103225319?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166047473116782388067276%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=166047473116782388067276&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-14-103225319-null-null.nonecase&utm_term=sklearn&spm=1018.2226.3001.4450
        #调参：https://www.bbsmax.com/A/ke5jrPKO5r/
        'random_state': [1],
        'n_jobs': [-1], # 报错

        'n_estimators': [500]#,
        # 'boosting_type':['gbdt'],#提升树的类型 gbdt,dart,goss,rf
        # 'num_leavel':[32],#树的最大叶子数，需小于2^(max_depth)
        # 'max_depth': [-1,5],#-1表示最大深度
        # 'early_stopping_rounds': [100,150],
        # 'learning_rate': [0.1, 0.05],
        # 'objective': ['binary'],#binary
        # 'num_class': [0],#binary--0
        # 'min_child_weight': [0.01, 0.001],#分支结点的最小权重
        # 'subsample': [0.8, 0.9, 1],
        # 'colsample_bytree': [0.8],#训练特征采样率 列
        # 'reg_alpha': [0,0.5,1],
        # 'reg_lambda': [0,0.5,1],
        # 'min_split_gain': [0],#最小分割增益
        # 'min_child_samples': [20, 30]
    },
    params_knn={#已解决
        'weights': ['uniform', 'distance'],
        'n_neighbors': [i for i in range(1, 5)],
        'p': [i for i in range(1, 3)],#控制Minkowski度量方法的值,整型，默认为2
        'n_jobs': [-1],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']#计算n近邻的算法
    },
    params_catboost={
        'random_seed': [1],
        'thread_count': [-1],#调用所有核心
        'task_type': ['GPU'],#调用GPU

        'depth': [10],#树深，最大16，建议在1到10之间。默认6
        'learning_rate': [0.03],#学习率默认0.03
        # 'l2_leaf_reg': [1, 2, 3, 4],#正则参数。默认3
        'iterations': [900],  # iterations 最大树数。默认1000
        # 'sampling_frequency': ['PerTree', 'PerTreeLevel'],#设置创建树时的采样频率，可选值PerTree/PerTreeLevel，默认为PerTreeLevel
        # 'random_leaf_estimation_method': ['Newton', 'Gradient'],#计算叶子值的方法，Newton/ Gradient。默认Gradient。
        #
        # # 'class_weights': [None],#类别的权重。默认None。
        # # 'class_scale_pos_weight': [None],  #二进制分类中class 1的权重。该值用作class 1中对象权重的乘数
        # 'approx_on_full_history': [False, True],# 计算近似值，False：使用1／fold_len_multiplier计算；True：使用fold中前面所有行计算。默认False。
        # 'random_strength': [0.2, 0.5, 1],#分数标准差乘数。默认1。
        # 'bootstrap_type': ['Bayesian', 'Bernoulli', 'No'],#定义权重计算逻辑，默认为Bayesian
        # 'bagging_temperature': [0.2, 0.5, 1]#贝叶斯套袋控制强度，区间[0, 1]。默认1
    }
)





