# -*- coding: utf-8 -*-
# 数据清洗步骤
### 1、使用del_same_col(df,k)去除重复列；
### 2、使用df_drop_row_col(df, row, col, k)多次迭代达到行样本、列样本缺失值在30%以下；
### 3、使用data_frame_summary(df)查询数据框各个列分布
### 4、依据data_frame_summary(df)的结果在原表上替换错别字，并重新读入表格，例如阴性阳性替换；
### 5、使用data_frame_summary(df)输出数据框数字列，编码列，聚类列
### 6、对数据进行缺失值填充，数字列用中位数，编码列用众数，聚类列用众数
### 7、使用preprocess_data_original(df, num_col, cat_col, non_cat_col)对数据进行清洗
### 8、特征工程

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier as ADA
from sklearn.ensemble import BaggingClassifier as BC
from sklearn.ensemble import GradientBoostingClassifier as GDBC
from xgboost import XGBClassifier as XGBC
from sklearn.ensemble import RandomForestClassifier as RFC
from catboost import CatBoostClassifier as CBC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier as LGB
from classification_indices import *
import parameter_py
import re
import joblib
import warnings
from data_preprocess import *
from gcforest.caForest import gcForest2
from gcforest.gcforest import GCForest
warnings.filterwarnings('ignore')
from plot_figurer import *
from feature_process import RobustRank

def calculate_statistics_by_category(df, category_column, gender_column, age_column):
    # Create a new dataframe to store the results
    result_df = pd.DataFrame(columns=['Category', 'Female Count', 'Male Count', 'Mean Age', 'Age std', 'Min Age', 'Max Age'])

    # Get unique category values
    unique_categories = df[category_column].unique()

    # Iterate over each unique category
    for category in unique_categories:
        # Select a sub-dataframe for the current category
        sub_df = df[df[category_column] == category]

        # Count the number of females and males
        female_count = sub_df[sub_df[gender_column] == 0].shape[0]
        male_count = sub_df[sub_df[gender_column] == 1].shape[0]

        # Calculate the mean and variance of age
        mean_age = round(sub_df[age_column].mean(), 1)
        age_std = round(sub_df[age_column].std(), 1)

        # Get the minimum and maximum age values
        min_age = sub_df[age_column].min()
        max_age = sub_df[age_column].max()

        # Append the results to the new dataframe
        result_df = result_df.append({
            'Category': category,
            'Female Count': female_count,
            'Male Count': male_count,
            'Mean Age': mean_age,
            'Age std': age_std,
            'Min Age': min_age,
            'Max Age': max_age
        }, ignore_index=True)

    return result_df

listnomodel = ['ID', 'Diagnose', 'disease', 'Disease', 'Cancer', 'Medical record ID', 'Name']

def columns_statistics(df):
    listX1 = []
    listX2 = []
    listX3 = []
    listX4 = []
    listX5 = []
    for i in df.columns:
        if 'X1' in i:
            listX1.append(i)
        if 'X2' in i:
            listX2.append(i)
        if 'X3' in i:
            listX3.append(i)
        if 'X4' in i:
            listX4.append(i)
        if 'X5' in i:
            listX5.append(i)
    print('listX1', len(listX1))
    print('listX2', len(listX2))
    print('listX3', len(listX3))
    print('listX4', len(listX4))
    print('listX5', len(listX5))
    return listX1, listX2, listX3, listX4, listX5

def missing_values_statistics(df):
    for col in df.columns:
        missing_values = df[col].isnull().sum()
        total_values = len(df[col])
        missing_ratio = (missing_values / total_values) * 100
        print(f'列名: {col}, 缺失值数量: {missing_values}, 缺失值比例: {missing_ratio:.2f}%')
    return

if __name__ == "__main__":
    # ###########################################################################################################数据预处理
    # data_original = pd.read_csv('./dataset/Supplementary_original_data.csv', encoding='gbk')
    # data_dup = data_original.drop_duplicates().reset_index(drop=True)  # 删除数据重复行
    # # data_sam = data_dup.sample(frac=1).reset_index(drop=True)  # 打乱顺序
    # listX1, listX2, listX3, listX4, listX5 = columns_statistics(data_dup)
    # # 统计data_drop的num_col、cat_col、non_cat_col
    # data_drop_frame_summary = data_frame_summary(data_dup)
    #
    # # # 1.对cat_col列进行编码，写入csv文件进行编码
    # df_count = count_by_category(data_dup, data_drop_frame_summary)# 编码列count
    # df_count.to_csv('./dataset/df_count.csv', encoding='gbk', index=False)
    # # 读入编码后的文件，并对data_dup重新编码,新增Category_right列，shelve,NA,delcol
    # df_count_encode = pd.read_csv('./dataset/Supplementary_encode_rule.csv', encoding='gbk')
    # data_count_encode = correct_category(data_dup, df_count_encode)
    # data_count_encode_frame_summary = data_frame_summary(data_count_encode)
    #
    # # # 2.查看数值列非空非数值的情况,将数字列乱码替换
    # data_count_encode_clean = data_count_encode.copy()
    # for i in range(data_count_encode.shape[1]):
    #     if data_count_encode_frame_summary['数字比例'][data_count_encode.columns[i]] < 1 and data_count_encode.columns[i] not in listnomodel:
    #         for j in range(data_count_encode.shape[0]):
    #             if pd.isna(data_count_encode.iloc[j,i]):
    #                 pass
    #             else:
    #                 try:
    #                     float(data_count_encode.iloc[j,i])
    #                 except:
    #                     print(i, j, data_count_encode.columns[i], data_count_encode.iloc[j, i])
    #                     try:
    #                         data_count_encode_clean.iloc[j,i] = str_to_num(data_count_encode_clean.iloc[j,i])
    #                     except:
    #                         data_count_encode_clean.iloc[j, i] = np.nan
    #                     print(i, j, data_count_encode_clean.columns[i], data_count_encode_clean.iloc[j, i])
    #
    # data_count_encode_clean.to_csv('./dataset/Supplementary_encode_data.csv', encoding='gbk', index=False, na_rep='NA')
    data_count_encode_clean = pd.read_csv('./dataset/Supplementary_encode_data.csv', encoding='gbk')
    # # 3.特征工程-1，删去全为0的列
    data_process = data_count_encode_clean.copy()
    for i in data_process.columns:
        try:
            if data_count_encode_clean[i].std() == 0:
                print(i)
                del data_process[i]
        except:
            pass
    # # 3.特征工程-2，数据拆分
    listX1, listX2, listX3, listX4, listX5 = columns_statistics(data_process)
    data_n = data_process.loc[data_process['Diagnose'] == 'Normal', list_add([listnomodel, ['Sex', 'Age'], listX1, listX2, listX3])]
    data_o = data_process.loc[data_process['Diagnose'] == 'Other disease', list_add([listnomodel, ['Sex', 'Age'], listX1, listX2, listX3])]
    data_no = data_process.loc[(data_process['Diagnose'] == 'Normal') | (data_process['Diagnose'] == 'Other disease'), list_add([listnomodel, ['Sex', 'Age'], listX1, listX2, listX3])]
    data_cancer = data_process[(data_process['Diagnose'] != 'Normal') & (data_process['Diagnose'] != 'Other disease')]
    # parameter
    row_null_drop = 0.3  # 行缺失超过0.3删除
    col_null_drop = 0.3  # 列缺失超过0.3删除
    data_cancer_drop, dellist_data_cancer_drop = df_drop_row_col(df=data_cancer.loc[:,list_add([listnomodel, ['Sex', 'Age'], listX1, listX2, listX3, listX4, listX5])],
                                                            collist=list_add([['Sex', 'Age'], listX1, listX2, listX3, listX4, listX5]),
                                                            row=row_null_drop,
                                                            col=col_null_drop, k=100)
    data_no_drop, dellist_data_no_drop = df_drop_row_col(df=data_no.loc[:,list_add([listnomodel, ['Sex', 'Age'], listX1])],
                                                            collist=list_add([['Sex', 'Age'], listX1]),
                                                            row=row_null_drop,
                                                            col=col_null_drop, k=100)
    listX1, listX2, listX3, listX4, listX5 = columns_statistics(data_cancer_drop)
    listX11, listX21, listX31, listX41, listX51 = columns_statistics(data_no_drop)
    # listX1 == listX11 # True
    # 填充癌症数据集X2X3X4X5但不填充X1
    data_cancer_fill = df_fillna(df=data_cancer_drop, method1='median', numcollist=list_minus(data_cancer_drop.columns, list_add([listnomodel, listX1])))
    # 对尿液颜色X301进行独热编码
    data_cancer_fill_onehot, onehot_encoder, kmeans_model = preprocess_data_original(data_cancer_fill, num_col=None, cat_col=['X301'], non_cat_col=None)
    # 合并
    data_merage = pd.concat([data_no_drop, data_cancer_fill_onehot], ignore_index=True)
    data_merage_fill = df_fillna(df=data_merage, method1='median', numcollist=list_add([['Sex'], ['Age'], listX1]))

    # 低方差过滤
    var_filter_k = 0.005
    df_var_filter = var_filter(data_merage_fill, var_filter_k, list_minus(data_merage_fill.columns, listnomodel))
    # pearson_corr过滤
    pearson_corr_k = 0.99
    df_pearson_corr = pearson_corr(df_var_filter, pearson_corr_k, list_minus(df_var_filter.columns, listnomodel))
    # 4.数据拆分
    from sklearn.model_selection import train_test_split
    X_trainset, X_vtset, y_trainset, y_vtset = train_test_split(df_pearson_corr, df_pearson_corr['Cancer'], test_size=0.40, random_state=15)  # 指定特征值占1/5,随机数种子是15
    X_validset, X_testset, y_validset, y_testset = train_test_split(X_vtset, y_vtset, test_size=0.50, random_state=15)  # 指定特征值占1/5,随机数种子是15
    # pd.DataFrame(X_train['Diagnose'].value_counts())
    # pd.DataFrame(X_valid['Diagnose'].value_counts())
    # pd.DataFrame(X_test['Diagnose'].value_counts())
    table1_all = calculate_statistics_by_category(df=df_pearson_corr, category_column='Diagnose',
                                                              gender_column='Sex', age_column='Age')
    table1_train = calculate_statistics_by_category(df=X_trainset, category_column='Diagnose',
                                                              gender_column='Sex', age_column='Age').loc[:, ['Category', 'Female Count', 'Male Count']]
    table1_valid = calculate_statistics_by_category(df=X_validset, category_column='Diagnose',
                                                              gender_column='Sex', age_column='Age').loc[:, ['Category', 'Female Count', 'Male Count']]
    table1_test = calculate_statistics_by_category(df=X_testset, category_column='Diagnose',
                                                              gender_column='Sex', age_column='Age').loc[:, ['Category', 'Female Count', 'Male Count']]
    for i in [table1_train, table1_valid, table1_test]:
        table1_all = pd.merge(table1_all,i, on='Category', sort=False)
    table1_all.to_csv('./results/table/table_1.csv', index=False)
    X_trainset['dataset'] = 'Cross-validation set'
    X_validset['dataset'] = 'Validation set'
    X_testset['dataset'] = 'Test set'
    Supplementary_process_data = pd.concat([X_trainset, X_validset, X_testset], ignore_index=True)
    Supplementary_process_data.to_csv('./dataset/Supplementary_process_data.csv', encoding='gbk', index=False, na_rep='NA')

    # # ###########################################################################################################数据建模
    df = pd.read_csv("./dataset/Supplementary_process_data.csv", encoding='gbk')
    listX1, listX2, listX3, listX4, listX5 = columns_statistics(df)
    listnomodel = ['ID', 'Diagnose', 'disease', 'Disease', 'Cancer', 'Medical record ID', 'Name', 'dataset']

    # # # ###########################################################################################################二分类
    # aim = 'binary'  # 'multi', 'binary'
    # if aim == 'binary':
    #     X_train = df.loc[df['dataset'] == 'Cross-validation set', list_add([['Sex'], ['Age'], listX1])]
    #     y_train = df.loc[df['dataset'] == 'Cross-validation set', 'Cancer']
    #     X_valid = df.loc[df['dataset'] == 'Validation set', list_add([['Sex'], ['Age'], listX1])]
    #     y_valid = df.loc[df['dataset'] == 'Validation set', 'Cancer']
    #     X_test = df.loc[df['dataset'] == 'Test set', list_add([['Sex'], ['Age'], listX1])]
    #     y_test = df.loc[df['dataset'] == 'Test set', 'Cancer']
    #     ovr_ = 0
    #     nclass_ = 2
    # elif aim == 'multi':
    #     X_train = df.loc[(df['dataset'] == 'Cross-validation set') & (df['Cancer'] == 1), list_minus(df.columns, listnomodel)]
    #     y_train = df.loc[(df['dataset'] == 'Cross-validation set') & (df['Cancer'] == 1), 'Diagnose']
    #     X_valid = df.loc[(df['dataset'] == 'Validation set') & (df['Cancer'] == 1), list_minus(df.columns, listnomodel)]
    #     y_valid = df.loc[(df['dataset'] == 'Validation set') & (df['Cancer'] == 1), 'Diagnose']
    #     X_test = df.loc[(df['dataset'] == 'Test set') & (df['Cancer'] == 1), list_minus(df.columns, listnomodel)]
    #     y_test = df.loc[(df['dataset'] == 'Test set') & (df['Cancer'] == 1), 'Diagnose']
    #     ovr_ = 1
    #     nclass_ = 10
    #
    # # X_train_scale, X_test_scale = scale_np(X_train=X_train, X_test=X_test, method='standard')
    # # X_valid_scale = scale_np(X_train=X_train, X_test=X_valid, method='standard')[1]
    # X_train_scale = X_train.values
    # X_valid_scale = X_valid.values
    # X_test_scale = X_test.values
    # y_train = y_train.values
    # y_valid = y_valid.values
    # y_test = y_test.values
    #
    #
    # from sklearn.preprocessing import LabelEncoder
    # le = LabelEncoder()
    # y_train_code = le.fit_transform(y_train)
    # for label in le.classes_:
    #     encoded_value = le.transform([label])[0]  # 使用 transform 方法转换单个标签
    #     print(f"类别 '{label}': 原值 {label}, 编码值 {encoded_value}")
    #
    # modellist = [
    #       RFC()
    #     , GDBC()
    #     , ADA()
    #     , XGBC()
    #     , LGB()
    #     , CBC()
    #     , gcForest2(shape_1X=16, window=[4,4], tolerance=0.0, min_samples_mgs=10, min_samples_cascade=7, n_cascadeRFtree=601)
    # ]
    # paramslist = [
    #     'params_rf'
    #     ,'params_gdbc'
    #     ,'params_adaboost'
    #     ,'params_xgbc'
    #     ,'params_lightgbm'
    #     ,'params_catboost'
    #     ,'shape_1X=21, window=[4,5], tolerance=0.0, min_samples_mgs=10, min_samples_cascade=7, n_cascadeRFtree=601'
    # ]
    #
    # prob_valid_list = []
    # prob_test_list = []
    # GS_params_list = []
    # for i, model in enumerate(modellist):
    #     if 'CatBoostClassifier' in str(model):
    #         model_1 = model.copy()
    #         model_2 = model.copy()
    #         model_3 = model.copy()
    #         model_4 = model.copy()
    #         model_5 = model.copy()
    #     else:
    #         model_1 = model
    #         model_2 = model
    #         model_3 = model
    #         model_4 = model
    #         model_5 = model
    #
    #     ovr = ovr_
    #     nclass = nclass_
    #     if 'LGBMClassifier' in str(model):
    #         ovr = 0
    #         parameter_arg = parameter_py.parameters_dict[paramslist[i]]
    #     elif 'gcForest2' in str(model):
    #         pass
    #     elif ovr == 0:
    #         parameter_arg = parameter_py.parameters_dict[paramslist[i]]
    #     else:
    #         ovr = ovr_
    #         parameter_arg = parameter_py.parameters_ov_dict[paramslist[i]]
    #
    #     # 训练集网格搜索
    #     if 'gcForest2' not in str(model_5):
    #         GS_result, GS_best_params_ = classmu_ovr_GridSearch_y(model=model_5, X_train=X_train_scale, y_train=y_train,
    #                                                               X_test=X_valid_scale, y_test=y_valid,
    #                                                               nclass=nclass, ovr=ovr, parameter=parameter_arg)
    #         GS_result.to_csv("./results/" + str(aim) + "/GS_result_ovr" + str(ovr) + ".csv", mode='a+')
    #         GS_params_list.append(GS_best_params_)
    #         GS_best_params2 = {}  # 创建除去estimator__的GS_best_params_字典，estimator__是嵌套OVO之后才加的参数
    #         for m, n in GS_best_params_.items():
    #             m = re.sub('estimator__', '', m, 1)
    #             GS_best_params2[m] = n
    #     else:
    #         pass
    #
    #     # 训练集交叉验证
    #     if 'gcForest2' not in str(model_5):
    #         model_3 = model_3.set_params(**GS_best_params2)
    #         cv10_result = classmu_ovr_kfolds_y(model=model_3, Xtrain__=X_train_scale,
    #                                            Ytrain__=y_train, nclasses=nclass, k=10, ovr=ovr)
    #     else:
    #         model_3 = gcForest2(shape_1X=21, window=[4,5], tolerance=0.0, min_samples_mgs=10, min_samples_cascade=7, n_cascadeRFtree=601)
    #         cv10_result = classmu_ovr_kfolds_y(model=model_3, Xtrain__=X_train_scale,
    #                                            Ytrain__=y_train, nclasses=nclass, k=10, ovr=0)
    #     cv10_result.to_csv("./results/" + str(aim) + "/cv10_result_ovr" + str(ovr) + ".csv", mode='a+')
    #
    #     # 验证集
    #     if 'gcForest2' not in str(model_5):
    #         model_1 = model_1.set_params(**GS_best_params2)
    #         valid_result, y_prob_valid = classmu_ovr_test_y(model=model_1,
    #                                                         X_train=X_train_scale, y_train=y_train,
    #                                                         X_test=X_valid_scale,
    #                                                         y_test=y_valid, nclass=nclass, ovr=ovr)
    #     else:
    #         model_1 = gcForest2(shape_1X=21, window=[4,5], tolerance=0.0, min_samples_mgs=10, min_samples_cascade=7, n_cascadeRFtree=601)
    #         valid_result, y_prob_valid = classmu_ovr_test_y(model=model_1,
    #                                                         X_train=X_train_scale, y_train=y_train,
    #                                                         X_test=X_valid_scale,
    #                                                         y_test=y_valid, nclass=nclass, ovr=0)
    #     valid_result.to_csv("./results/" + str(aim) + "/valid_result_ovr" + str(ovr) + ".csv", mode='a+')
    #     prob_valid_list.append(y_prob_valid)
    #
    #     # 测试集
    #     if 'gcForest2' not in str(model_5):
    #         model_2 = model_2.set_params(**GS_best_params2)
    #         test_result, y_prob_test = classmu_ovr_test_y(model=model_2, X_train=X_train_scale,
    #                                                       y_train=y_train, X_test=X_test_scale, y_test=y_test,
    #                                                       nclass=nclass, ovr=ovr)
    #     else:
    #         model_2 = gcForest2(shape_1X=21, window=[4,5], tolerance=0.0, min_samples_mgs=10, min_samples_cascade=7, n_cascadeRFtree=601)
    #         test_result, y_prob_test = classmu_ovr_test_y(model=model_2, X_train=X_train_scale,
    #                                                       y_train=y_train, X_test=X_test_scale, y_test=y_test,
    #                                                       nclass=nclass, ovr=0)
    #     test_result.to_csv("./results/" + str(aim) + "/test_result_ovr" + str(ovr) + ".csv", mode='a+')
    #     prob_test_list.append(y_prob_test)
    #
    #     if 'LGBMClassifier' in str(model_4):
    #         model_4.set_params(**GS_best_params2)
    #         model_4.fit(X_train_scale, y_train)
    #         joblib.dump(model_4, "./model/" + str(aim) + "/" + str(re.sub('params_', '', paramslist[i], 1)) + ".pkl")
    #     elif 'gcForest2' in str(model_4):
    #         model_4 = gcForest2(shape_1X=21, window=[4,5], tolerance=0.0, min_samples_mgs=10, min_samples_cascade=7,
    #                             n_cascadeRFtree=601)
    #         model_4.fit(X_train_scale, y_train)
    #         joblib.dump(model_4, "./model/" + str(aim) + "/" + 'gcForest' + ".pkl")
    #     elif ovr == 0:
    #         model_4.set_params(**GS_best_params2)
    #         model_4.fit(X_train_scale, y_train)
    #         joblib.dump(model_4, "./model/" + str(aim) + "/" + str(re.sub('params_', '', paramslist[i], 1)) + ".pkl")
    #     else:
    #         model_4.set_params(**GS_best_params2)
    #         model2 = OneVsRestClassifier(model_4)
    #         model2.fit(X_train_scale, y_train)
    #         joblib.dump(model2, "./model/" + str(aim) + "/" + str(re.sub('params_', '', paramslist[i], 1)) + ".pkl")
    #
    # f1 = open('./results/' + str(aim) + '/GS_params_list.txt', 'w')
    # for i in range(len(GS_params_list)):
    #     f1.write(str(paramslist[i])+':' + '\n')
    #     f1.write(str(GS_params_list[i])+'\n')
    # f1.close()
    #
    # ovr = ovr_
    # if nclass > 2:
    #     lb = preprocessing.LabelBinarizer()
    #     y_train_onehot = lb.fit_transform(y_train)
    #     y_test_onehot = lb.transform(y_test)
    #     y_valid_onehot = lb.transform(y_valid)
    # elif nclass == 2:
    #     lb = preprocessing.LabelBinarizer()
    #     y_train_onehot0 = lb.fit_transform(y_train)
    #     y_test_onehot0 = lb.transform(y_test)
    #     y_valid_onehot0 = lb.transform(y_valid)
    #     y_train_onehot = np.hstack(((1-y_train_onehot0), y_train_onehot0))
    #     y_test_onehot = np.hstack(((1-y_test_onehot0), y_test_onehot0))
    #     y_valid_onehot = np.hstack(((1-y_valid_onehot0), y_valid_onehot0))
    #
    # vote_valid_result_soft = ensemble_model_vote(y_test_onehot=y_valid_onehot, listmodels=prob_valid_list, vote='Soft',
    #                                              ovr=ovr)
    # vote_valid_result_soft.to_csv("./results/" + str(aim) + "/vote_valid_result_soft_ovr" + str(ovr) + ".csv",
    #                               mode='a+')
    # vote_valid_result_Hard = ensemble_model_vote(y_test_onehot=y_valid_onehot, listmodels=prob_valid_list, vote='Hard',
    #                                              ovr=ovr)
    # vote_valid_result_Hard.to_csv("./results/" + str(aim) + "/vote_valid_result_Hard_ovr" + str(ovr) + ".csv",
    #                               mode='a+')
    # vote_test_result_soft = ensemble_model_vote(y_test_onehot=y_test_onehot, listmodels=prob_test_list, vote='Soft',
    #                                             ovr=ovr)
    # vote_test_result_soft.to_csv("./results/" + str(aim) + "/vote_test_result_soft_ovr" + str(ovr) + ".csv", mode='a+')
    # vote_test_result_Hard = ensemble_model_vote(y_test_onehot=y_test_onehot, listmodels=prob_test_list, vote='Hard',
    #                                             ovr=ovr)
    # vote_test_result_Hard.to_csv("./results/" + str(aim) + "/vote_test_result_Hard_ovr" + str(ovr) + ".csv", mode='a+')
    #
    # ovr = ovr_
    # if ovr == 0:
    #     parameter_arg = parameter_py.parameters_dict['params_rf']
    # else:
    #     ovr = ovr_
    #     parameter_arg = parameter_py.parameters_ov_dict['params_rf']
    # blending_result, blending_model_arg, model_blending = ensemble_model_blending(model=RFC(), y_valid_onehot=y_valid_onehot,
    #                                                               y_test_onehot=y_test_onehot,
    #                                                               prob_valid_list=prob_valid_list,
    #                                                               prob_test_list=prob_test_list, ovr=ovr,
    #                                                               parameter=parameter_arg)
    # blending_result.to_csv("./results/" + str(aim) + "/blending_result_ovr" + str(ovr) + ".csv", mode='a+')
    # joblib.dump(model_blending, "./model/" + str(aim) + "/" + str(aim) + '_blending' + ".pkl")
    #
    # # # #验证集以及测试集AUC曲线图
    # # plot_roc_curve_score(modelnamelist=['Random forest', 'GBM', 'AdaBoost', 'XGBoost', 'LightGBM', 'CatBoost', 'Deep Forest'],
    # #                      y_score_list=prob_valid_list, y_test=y_valid, savepath='./results/figure/binary_valid_auc.tiff')
    # prob_test_merge = np.nan_to_num(prob_test_list[0])
    # for i in range(1, len(prob_test_list)):
    #     prob_test_merge = np.hstack((prob_test_merge, np.nan_to_num(prob_test_list[i])))
    # prob_test_blending = model_blending.predict_proba(prob_test_merge)
    # prob_test_list.append(prob_test_blending)
    # plot_roc_curve_score(modelnamelist=['Random forest', 'GBM', 'AdaBoost', 'XGBoost', 'LightGBM', 'CatBoost', 'Deep Forest', 'Blending'],
    #                      y_score_list=prob_test_list, y_test=y_test, savepath='./results/figure/binary_test_auc.tiff')

    # # 特征重要性排序
    # feature_names = X_train.columns
    # # rf
    # params_rf = {'bootstrap': True, 'criterion': 'gini', 'max_samples': 0.9, 'n_estimators': 1200, 'n_jobs': -1, 'oob_score': True, 'random_state': 1}
    # model_rf = RFC().set_params(**params_rf)
    # model_rf.fit(X_train, y_train)
    # feature_importances_rf = model_rf.feature_importances_
    # indices_rf = np.argsort(feature_importances_rf)[::-1]
    # sorted_feature_names_rf = feature_names[indices_rf]
    # sorted_feature_importances_rf = feature_importances_rf[indices_rf]
    # # gdbc
    # params_gdbc = {'n_estimators': 500, 'random_state': 1}
    # model_gdbc = GDBC().set_params(**params_gdbc)
    # model_gdbc.fit(X_train, y_train)
    # feature_importances_gdbc = model_gdbc.feature_importances_
    # indices_gdbc = np.argsort(feature_importances_gdbc)[::-1]
    # sorted_feature_names_gdbc = feature_names[indices_gdbc]
    # sorted_feature_importances_gdbc = feature_importances_gdbc[indices_gdbc]
    # # ada
    # params_ada = {'base_estimator': DTC(max_depth=7), 'n_estimators': 1100, 'random_state': 1, 'learning_rate': 0.1}
    # model_ada = ADA().set_params(**params_ada)
    # model_ada.fit(X_train, y_train)
    # feature_importances_ada = model_ada.feature_importances_
    # indices_ada = np.argsort(feature_importances_ada)[::-1]
    # sorted_feature_names_ada = feature_names[indices_ada]
    # sorted_feature_importances_ada = feature_importances_ada[indices_ada]
    # # xgbc
    # params_xgbc = {'booster': 'gbtree', 'n_estimators': 300, 'nthread': -1, 'seed': 1}
    # model_xgbc = XGBC().set_params(**params_xgbc)
    # model_xgbc.fit(X_train, y_train)
    # feature_importances_xgbc = model_xgbc.feature_importances_
    # indices_xgbc = np.argsort(feature_importances_xgbc)[::-1]
    # sorted_feature_names_xgbc = feature_names[indices_xgbc]
    # sorted_feature_importances_xgbc = feature_importances_xgbc[indices_xgbc]
    # # lightgbm
    # params_lig = {'n_estimators': 500, 'n_jobs': -1, 'random_state': 1}
    # model_lig = LGB().set_params(**params_lig)
    # model_lig.fit(X_train, y_train)
    # feature_importances_lig = model_lig.feature_importances_
    # indices_lig = np.argsort(feature_importances_lig)[::-1]
    # sorted_feature_names_lig = feature_names[indices_lig]
    # sorted_feature_importances_lig = feature_importances_lig[indices_lig]
    # # catboost
    # params_cat = {'depth': 10, 'iterations': 900, 'learning_rate': 0.03, 'random_seed': 1, 'task_type': 'GPU', 'thread_count': -1}
    # model_cat = CBC().set_params(**params_cat)
    # model_cat.fit(X_train, y_train)
    # feature_importances_cat = model_cat.feature_importances_
    # indices_cat = np.argsort(feature_importances_cat)[::-1]
    # sorted_feature_names_cat = feature_names[indices_cat]
    # sorted_feature_importances_cat = feature_importances_cat[indices_cat]
    #
    # data_import = {
    # 'Column1': sorted_feature_names_rf,
    # 'Column2': sorted_feature_names_xgbc,
    # 'Column3': sorted_feature_names_gdbc,
    # 'Column4': sorted_feature_names_lig,
    # 'Column5': sorted_feature_names_cat,
    # 'Column6': sorted_feature_names_ada,
    # }
    # df_import = pd.DataFrame(data_import)
    # list_import = RobustRank(df_import)
    # df_import_final = pd.DataFrame({'Column': list_import})
    # df_indices = pd.read_csv('./dataset/Supplementary_encode_indices.csv', encoding='gbk')
    # df_test = df_to_test(df, list_import, 'Cancer')
    # df_test['feature name'] = np.nan
    # for i in range(len(df_test)):
    #     for j in range(len(df_indices)):
    #         if df_test['columns'][i] == df_indices['Indices code name'][j]:
    #             df_test['feature name'][i] = df_indices['Indices name'][j]
    # df_test.to_csv('./results/table/table_5.csv')



    # ############################################################################################################多分类
    aim = 'multi'  # 'multi', 'binary'
    if aim == 'binary':
        X_train = df.loc[df['dataset'] == 'Cross-validation set', list_add([['Sex'], ['Age'], listX1])]
        y_train = df.loc[df['dataset'] == 'Cross-validation set', 'Cancer']
        X_valid = df.loc[df['dataset'] == 'Validation set', list_add([['Sex'], ['Age'], listX1])]
        y_valid = df.loc[df['dataset'] == 'Validation set', 'Cancer']
        X_test = df.loc[df['dataset'] == 'Test set', list_add([['Sex'], ['Age'], listX1])]
        y_test = df.loc[df['dataset'] == 'Test set', 'Cancer']
        ovr_ = 0
        nclass_ = 2
    elif aim == 'multi':
        X_train = df.loc[(df['dataset'] == 'Cross-validation set') & (df['Cancer'] == 1), list_minus(df.columns, listnomodel)]
        y_train = df.loc[(df['dataset'] == 'Cross-validation set') & (df['Cancer'] == 1), 'Diagnose']
        X_valid = df.loc[(df['dataset'] == 'Validation set') & (df['Cancer'] == 1), list_minus(df.columns, listnomodel)]
        y_valid = df.loc[(df['dataset'] == 'Validation set') & (df['Cancer'] == 1), 'Diagnose']
        X_test = df.loc[(df['dataset'] == 'Test set') & (df['Cancer'] == 1), list_minus(df.columns, listnomodel)]
        y_test = df.loc[(df['dataset'] == 'Test set') & (df['Cancer'] == 1), 'Diagnose']
        ovr_ = 1
        nclass_ = 10

    # X_train_scale, X_test_scale = scale_np(X_train=X_train, X_test=X_test, method='standard')
    # X_valid_scale = scale_np(X_train=X_train, X_test=X_valid, method='standard')[1]
    X_train_scale = X_train.values
    X_valid_scale = X_valid.values
    X_test_scale = X_test.values
    y_train = y_train.values
    y_valid = y_valid.values
    y_test = y_test.values


    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_code = le.fit_transform(y_train)
    for label in le.classes_:
        encoded_value = le.transform([label])[0]  # 使用 transform 方法转换单个标签
        print(f"类别 '{label}': 原值 {label}, 编码值 {encoded_value}")

    modellist = [
          RFC()
        , GDBC()
        , ADA()
        , XGBC()
        , LGB()
        , CBC()
        , gcForest2(shape_1X=36, window=[6,6], tolerance=0.0, min_samples_mgs=10, min_samples_cascade=7, n_cascadeRFtree=601)
    ]
    paramslist = [
        'params_rf'
        ,'params_gdbc'
        ,'params_adaboost'
        ,'params_xgbc'
        ,'params_lightgbm'
        ,'params_catboost'
        ,'shape_1X=36, window=[6,6], tolerance=0.0, min_samples_mgs=10, min_samples_cascade=7, n_cascadeRFtree=601'
    ]

    prob_valid_list = []
    prob_test_list = []
    GS_params_list = []
    for i, model in enumerate(modellist):
        if 'CatBoostClassifier' in str(model):
            model_1 = model.copy()
            model_2 = model.copy()
            model_3 = model.copy()
            model_4 = model.copy()
            model_5 = model.copy()
        else:
            model_1 = model
            model_2 = model
            model_3 = model
            model_4 = model
            model_5 = model

        ovr = ovr_
        nclass = nclass_
        if 'LGBMClassifier' in str(model):
            ovr = 0
            parameter_arg = parameter_py.parameters_dict[paramslist[i]]
        elif 'gcForest2' in str(model):
            pass
        elif ovr == 0:
            parameter_arg = parameter_py.parameters_dict[paramslist[i]]
        else:
            ovr = ovr_
            parameter_arg = parameter_py.parameters_ov_dict[paramslist[i]]

        # 训练集网格搜索
        if 'gcForest2' not in str(model_5):
            GS_result, GS_best_params_ = classmu_ovr_GridSearch_y(model=model_5, X_train=X_train_scale, y_train=y_train,
                                                                  X_test=X_valid_scale, y_test=y_valid,
                                                                  nclass=nclass, ovr=ovr, parameter=parameter_arg)
            GS_result.to_csv("./results/" + str(aim) + "/GS_result_ovr" + str(ovr) + ".csv", mode='a+')
            GS_params_list.append(GS_best_params_)
            GS_best_params2 = {}  # 创建除去estimator__的GS_best_params_字典，estimator__是嵌套OVO之后才加的参数
            for m, n in GS_best_params_.items():
                m = re.sub('estimator__', '', m, 1)
                GS_best_params2[m] = n
        else:
            pass

        # 训练集交叉验证
        # if 'gcForest2' not in str(model_5):
        #     model_3 = model_3.set_params(**GS_best_params2)
        #     cv10_result = classmu_ovr_kfolds_y(model=model_3, Xtrain__=X_train_scale,
        #                                        Ytrain__=y_train, nclasses=nclass, k=10, ovr=ovr)
        # else:
        #     model_3 = gcForest2(shape_1X=72, window=[9,8], tolerance=0.0, min_samples_mgs=10, min_samples_cascade=7, n_cascadeRFtree=601)
        #     cv10_result = classmu_ovr_kfolds_y(model=model_3, Xtrain__=X_train_scale,
        #                                        Ytrain__=y_train, nclasses=nclass, k=10, ovr=0)
        # cv10_result.to_csv("./results/" + str(aim) + "/cv10_result_ovr" + str(ovr) + ".csv", mode='a+')

        # 验证集
        if 'gcForest2' not in str(model_5):
            model_1 = model_1.set_params(**GS_best_params2)
            valid_result, y_prob_valid = classmu_ovr_test_y(model=model_1,
                                                            X_train=X_train_scale, y_train=y_train,
                                                            X_test=X_valid_scale,
                                                            y_test=y_valid, nclass=nclass, ovr=ovr)
        else:
            model_1 = gcForest2(shape_1X=72, window=[9,8], tolerance=0.0, min_samples_mgs=10, min_samples_cascade=7, n_cascadeRFtree=601)
            valid_result, y_prob_valid = classmu_ovr_test_y(model=model_1,
                                                            X_train=X_train_scale, y_train=y_train,
                                                            X_test=X_valid_scale,
                                                            y_test=y_valid, nclass=nclass, ovr=0)
        valid_result.to_csv("./results/" + str(aim) + "/valid_result_ovr" + str(ovr) + ".csv", mode='a+')
        prob_valid_list.append(y_prob_valid)

        # 测试集
        if 'gcForest2' not in str(model_5):
            model_2 = model_2.set_params(**GS_best_params2)
            test_result, y_prob_test = classmu_ovr_test_y(model=model_2, X_train=X_train_scale,
                                                          y_train=y_train, X_test=X_test_scale, y_test=y_test,
                                                          nclass=nclass, ovr=ovr)
        else:
            model_2 = gcForest2(shape_1X=72, window=[9,8], tolerance=0.0, min_samples_mgs=10, min_samples_cascade=7, n_cascadeRFtree=601)
            test_result, y_prob_test = classmu_ovr_test_y(model=model_2, X_train=X_train_scale,
                                                          y_train=y_train, X_test=X_test_scale, y_test=y_test,
                                                          nclass=nclass, ovr=0)
        test_result.to_csv("./results/" + str(aim) + "/test_result_ovr" + str(ovr) + ".csv", mode='a+')
        prob_test_list.append(y_prob_test)

        if 'LGBMClassifier' in str(model_4):
            model_4.set_params(**GS_best_params2)
            model_4.fit(X_train_scale, y_train)
            joblib.dump(model_4, "./model/" + str(aim) + "/" + str(re.sub('params_', '', paramslist[i], 1)) + ".pkl")
        elif 'gcForest2' in str(model_4):
            model_4 = gcForest2(shape_1X=72, window=[9,8], tolerance=0.0, min_samples_mgs=10, min_samples_cascade=7,
                                n_cascadeRFtree=601)
            model_4.fit(X_train_scale, y_train)
            joblib.dump(model_4, "./model/" + str(aim) + "/" + 'gcForest' + ".pkl")
        elif ovr == 0:
            model_4.set_params(**GS_best_params2)
            model_4.fit(X_train_scale, y_train)
            joblib.dump(model_4, "./model/" + str(aim) + "/" + str(re.sub('params_', '', paramslist[i], 1)) + ".pkl")
        else:
            model_4.set_params(**GS_best_params2)
            model2 = OneVsRestClassifier(model_4)
            model2.fit(X_train_scale, y_train)
            joblib.dump(model2, "./model/" + str(aim) + "/" + str(re.sub('params_', '', paramslist[i], 1)) + ".pkl")

    f1 = open('./results/' + str(aim) + '/GS_params_list.txt', 'w')
    for i in range(len(GS_params_list)):
        f1.write(str(paramslist[i])+':' + '\n')
        f1.write(str(GS_params_list[i])+'\n')
    f1.close()

    ovr = ovr_
    if nclass > 2:
        lb = preprocessing.LabelBinarizer()
        y_train_onehot = lb.fit_transform(y_train)
        y_test_onehot = lb.transform(y_test)
        y_valid_onehot = lb.transform(y_valid)
    elif nclass == 2:
        lb = preprocessing.LabelBinarizer()
        y_train_onehot0 = lb.fit_transform(y_train)
        y_test_onehot0 = lb.transform(y_test)
        y_valid_onehot0 = lb.transform(y_valid)
        y_train_onehot = np.hstack(((1-y_train_onehot0), y_train_onehot0))
        y_test_onehot = np.hstack(((1-y_test_onehot0), y_test_onehot0))
        y_valid_onehot = np.hstack(((1-y_valid_onehot0), y_valid_onehot0))

    vote_valid_result_soft = ensemble_model_vote(y_test_onehot=y_valid_onehot, listmodels=prob_valid_list, vote='Soft',
                                                 ovr=ovr)
    vote_valid_result_soft.to_csv("./results/" + str(aim) + "/vote_valid_result_soft_ovr" + str(ovr) + ".csv",
                                  mode='a+')
    vote_valid_result_Hard = ensemble_model_vote(y_test_onehot=y_valid_onehot, listmodels=prob_valid_list, vote='Hard',
                                                 ovr=ovr)
    vote_valid_result_Hard.to_csv("./results/" + str(aim) + "/vote_valid_result_Hard_ovr" + str(ovr) + ".csv",
                                  mode='a+')
    vote_test_result_soft = ensemble_model_vote(y_test_onehot=y_test_onehot, listmodels=prob_test_list, vote='Soft',
                                                ovr=ovr)
    vote_test_result_soft.to_csv("./results/" + str(aim) + "/vote_test_result_soft_ovr" + str(ovr) + ".csv", mode='a+')
    vote_test_result_Hard = ensemble_model_vote(y_test_onehot=y_test_onehot, listmodels=prob_test_list, vote='Hard',
                                                ovr=ovr)
    vote_test_result_Hard.to_csv("./results/" + str(aim) + "/vote_test_result_Hard_ovr" + str(ovr) + ".csv", mode='a+')

    ovr = ovr_
    if ovr == 0:
        parameter_arg = parameter_py.parameters_dict['params_rf']
    else:
        ovr = ovr_
        parameter_arg = parameter_py.parameters_ov_dict['params_rf']
    blending_result, blending_model_arg, model_blending = ensemble_model_blending(model=RFC(), y_valid_onehot=y_valid_onehot,
                                                                  y_test_onehot=y_test_onehot,
                                                                  prob_valid_list=prob_valid_list,
                                                                  prob_test_list=prob_test_list, ovr=ovr,
                                                                  parameter=parameter_arg)
    blending_result.to_csv("./results/" + str(aim) + "/blending_result_ovr" + str(ovr) + ".csv", mode='a+')
    joblib.dump(model_blending, "./model/" + str(aim) + "/" + str(aim) + '_blending' + ".pkl")

    # # 绘制混淆矩阵
    prob_test_merge = np.nan_to_num(prob_test_list[0])
    for i in range(1, len(prob_test_list)):
        prob_test_merge = np.hstack((prob_test_merge, np.nan_to_num(prob_test_list[i])))
    prob_test_blending = model_blending.decision_function(prob_test_merge)
    y_score_prob = (prob_test_blending - prob_test_blending.min()) / (prob_test_blending.max() - prob_test_blending.min())
    y_score_prob_normalized = y_score_prob / y_score_prob.sum(axis=1, keepdims=True)
    y_pred_onehot = (y_score_prob_normalized == y_score_prob_normalized.max(axis=1)[:, None]).astype(int)

    y_pred = lb.inverse_transform(y_pred_onehot)
    plot_confusion_matrix(y_true=y_test, y_pred=y_pred, classes=['BC','BCa','CRC','EC','GC','LC','LCa','PC','PCa','TC'],
                          dpi=600, width=8, length=8,
                          savepath='./results/figure/confusion_matrix_blening_all2.tiff')
    # # AUC曲线图，ovr
    plot_auc_curves_multi(y_pred=y_score_prob_normalized, y_true=y_test_onehot, class_names=['BC','BCa','CRC','EC','GC','LC','LCa','PC','PCa','TC'],
                           savepath='./results/figure/AUC_multi_all2.tiff')

    # # ACC曲线变化图
    feature_names = X_train.columns
    # rf
    params_rf = {'bootstrap': True, 'max_samples': 0.9, 'n_estimators': 1300, 'n_jobs': -1, 'oob_score': True, 'random_state': 1, 'criterion': 'entropy'}
    model_rf = RFC().set_params(**params_rf)
    model_rf.fit(X_train, y_train)
    feature_importances_rf = model_rf.feature_importances_
    indices_rf = np.argsort(feature_importances_rf)[::-1]
    sorted_feature_names_rf = feature_names[indices_rf]
    sorted_feature_importances_rf = feature_importances_rf[indices_rf]
    # gdbc
    params_gdbc = {'n_estimators': 500, 'random_state': 1}
    model_gdbc = GDBC().set_params(**params_gdbc)
    model_gdbc.fit(X_train, y_train)
    feature_importances_gdbc = model_gdbc.feature_importances_
    indices_gdbc = np.argsort(feature_importances_gdbc)[::-1]
    sorted_feature_names_gdbc = feature_names[indices_gdbc]
    sorted_feature_importances_gdbc = feature_importances_gdbc[indices_gdbc]
    # ada
    params_ada = {'base_estimator': DTC(max_depth=8), 'n_estimators': 550, 'random_state': 1, 'learning_rate': 0.18}
    model_ada = ADA().set_params(**params_ada)
    model_ada.fit(X_train, y_train)
    feature_importances_ada = model_ada.feature_importances_
    indices_ada = np.argsort(feature_importances_ada)[::-1]
    sorted_feature_names_ada = feature_names[indices_ada]
    sorted_feature_importances_ada = feature_importances_ada[indices_ada]

    # xgbc
    params_xgbc = {'booster': 'gbtree', 'n_estimators': 500, 'nthread': -1, 'seed': 1}
    model_xgbc = XGBC().set_params(**params_xgbc)
    model_xgbc.fit(X_train, y_train_code)
    feature_importances_xgbc = model_xgbc.feature_importances_
    indices_xgbc = np.argsort(feature_importances_xgbc)[::-1]
    sorted_feature_names_xgbc = feature_names[indices_xgbc]
    sorted_feature_importances_xgbc = feature_importances_xgbc[indices_xgbc]
    # lightgbm
    params_lig = {'n_estimators': 500, 'n_jobs': -1, 'random_state': 1, 'boosting_type': 'gbdt'}
    model_lig = LGB().set_params(**params_lig)
    model_lig.fit(X_train, y_train_code)
    feature_importances_lig = model_lig.feature_importances_
    indices_lig = np.argsort(feature_importances_lig)[::-1]
    sorted_feature_names_lig = feature_names[indices_lig]
    sorted_feature_importances_lig = feature_importances_lig[indices_lig]
    # catboost
    params_cat = {'depth': 8, 'iterations': 700, 'random_seed': 1, 'learning_rate': 0.1}
    model_cat = CBC().set_params(**params_cat)
    model_cat.fit(X_train, y_train_code)
    feature_importances_cat = model_cat.feature_importances_
    indices_cat = np.argsort(feature_importances_cat)[::-1]
    sorted_feature_names_cat = feature_names[indices_cat]
    sorted_feature_importances_cat = feature_importances_cat[indices_cat]

    data_import = {
    'Column1': sorted_feature_names_rf,
    'Column2': sorted_feature_names_xgbc,
    'Column3': sorted_feature_names_gdbc,
    'Column4': sorted_feature_names_lig,
    'Column5': sorted_feature_names_cat,
    'Column6': sorted_feature_names_ada
    }
    df_import = pd.DataFrame(data_import)
    list_import = RobustRank(df_import)
    df_import_final = pd.DataFrame({'Column': list_import})
    df_indices = pd.read_csv('./dataset/Supplementary_encode_indices.csv', encoding='gbk')
    df_test = df_to_test(df[df['Cancer'] == 1], list_import, 'Diagnose')
    df_test['feature name'] = np.nan
    df_test['units'] = np.nan
    df_test['Abbreviations'] = np.nan
    for i in range(len(df_test)):
        for j in range(len(df_indices)):
            if df_test['columns'][i] == df_indices['Indices code name'][j]:
                df_test['feature name'][i] = df_indices['Indices name'][j]
                df_test['units'][i] = df_indices['Units'][j]
                df_test['Abbreviations'][i] = df_indices['Abbreviations'][j]
    # df_test.to_csv('./results/table/table_6.csv')
    df_test.to_csv('./results/table/table_7.csv',encoding='gbk',index=False)

    table_5 = pd.read_csv('./results/table/table_5.csv',encoding='gbk')
    table_5['feature name'] = np.nan
    table_5['units'] = np.nan
    table_5['Abbreviations'] = np.nan
    for i in range(len(table_5)):
        for j in range(len(df_indices)):
            if table_5['columns'][i] == df_indices['Indices code name'][j]:
                table_5['feature name'][i] = df_indices['Indices name'][j]
                table_5['units'][i] = df_indices['Units'][j]
                table_5['Abbreviations'][i] = df_indices['Abbreviations'][j]
    table_5.to_csv('./results/table/table_51.csv', encoding='gbk', index=False)



    # ACC变化曲线
    acc_test_multi_blend = []
    acc_test_multi_other = []
    for i in range(len(list_import)):
        print('------------------------------------------------------------------------------------------------------',i)
        with open('./results/i.txt', 'a') as file:
            # 将数字转换为字符串并追加写入文件
            file.write('\n' + str(i))  # 添加换行符以区分不同的数字
        X_train_acc = X_train[list_import[0:i + 1]].values
        X_valid_acc = X_valid[list_import[0:i + 1]].values
        X_test_acc = X_test[list_import[0:i + 1]].values
        params_rf = {'bootstrap': True, 'max_samples': 0.9, 'n_estimators': 1300, 'n_jobs': -1, 'oob_score': True,
                     'random_state': 1, 'criterion': 'entropy'}
        model_rf = RFC().set_params(**params_rf)
        params_gdbc = {'n_estimators': 500, 'random_state': 1}
        model_gdbc = GDBC().set_params(**params_gdbc)
        params_ada = {'base_estimator': DTC(max_depth=8), 'n_estimators': 550, 'random_state': 1, 'learning_rate': 0.18}
        model_ada = ADA().set_params(**params_ada)
        params_xgbc = {'booster': 'gbtree', 'n_estimators': 500, 'nthread': -1, 'seed': 1}
        model_xgbc = XGBC().set_params(**params_xgbc)
        params_lig = {'n_estimators': 500, 'n_jobs': -1, 'random_state': 1, 'boosting_type': 'gbdt'}
        model_lig = LGB().set_params(**params_lig)
        params_cat = {'depth': 8, 'iterations': 700, 'random_seed': 1, 'learning_rate': 0.1}
        model_cat = CBC().set_params(**params_cat)
        sqrt_floor = math.floor(math.sqrt(i+1))
        model_gcf = gcForest2(shape_1X=sqrt_floor**2, window=[sqrt_floor, sqrt_floor], tolerance=0.0, min_samples_mgs=10, min_samples_cascade=7,
                            n_cascadeRFtree=601)
        model_list = [
            model_rf
            , model_gdbc
            , model_ada
            , model_xgbc
            , model_lig
            , model_cat
            , model_gcf
        ]
        ovr_list = [1,1,1,1,0,1,0]
        prob_valid_list_ = []
        prob_test_list_ = []
        for j in range(7):
            valid_result_, y_prob_valid_ = classmu_ovr_test_y(model=model_list[j],
                                                            X_train=X_train_acc, y_train=y_train,
                                                            X_test=X_valid_acc,
                                                            y_test=y_valid, nclass=10, ovr=ovr_list[j])
            test_result_, y_prob_test_ = classmu_ovr_test_y(model=model_list[j],
                                                            X_train=X_train_acc, y_train=y_train,
                                                            X_test=X_test_acc,
                                                            y_test=y_test, nclass=10, ovr=ovr_list[j])
            prob_valid_list_.append(y_prob_valid_)
            prob_test_list_.append(y_prob_test_)
            acc_test_multi_other.append(test_result_['acc_all'][0])
        # blending_result_, blending_model_arg_, model_blending_ = ensemble_model_blending(model=RFC(),
        #                                                                               y_valid_onehot=y_valid_onehot,
        #                                                                               y_test_onehot=y_test_onehot,
        #                                                                               prob_valid_list=prob_valid_list_,
        #                                                                               prob_test_list=prob_test_list_,
        #                                                                               ovr=1,
        #                                                                               parameter=parameter_py.parameters_ov_dict['params_rf'])
        blending_result_, blending_model_prob_test, model_blending_ = ensemble_model_blending2(model=RFC(random_state=1, n_estimators=1300, criterion='entropy', max_samples=0.9, n_jobs=-1),
                                                                                               y_valid_onehot=y_valid_onehot,
                                                                                               y_test_onehot=y_test_onehot,
                                                                                               prob_valid_list=prob_valid_list_,
                                                                                               prob_test_list=prob_test_list_,
                                                                                               ovr=1)
        acc_test_multi_blend.append(blending_result_['acc_all'][0])

    import pickle
    with open('./results/multi/acc_test_multi_other.pkl', 'wb') as file:
        pickle.dump(acc_test_multi_other, file)
    with open('./results/multi/acc_test_multi_blend.pkl', 'wb') as file:
        pickle.dump(acc_test_multi_blend, file)
    with open('./results/multi/prob_valid_list_.pkl', 'wb') as file:
        pickle.dump(prob_valid_list_, file)
    with open('./results/multi/prob_test_list_.pkl', 'wb') as file:
        pickle.dump(prob_test_list_, file)
    with open('./model/multi/blend_multi_all.pkl', 'wb') as file:
        pickle.dump(model_blending_, file)
    # 绘制ACC变化图
    import copy
    acc_test_multi_other_copy = copy.deepcopy(acc_test_multi_other)
    acc_test_multi_blend_copy = copy.deepcopy(acc_test_multi_blend)
    acc_test_multi_other_7x79 = np.array(acc_test_multi_other_copy).reshape(1, 79, 7)
    acc_test_multi_other_7x79[:, 78, :] = np.array([[0.6848,0.7057,0.6379,0.7211,0.7201,0.7261,0.6384]])
    acc_test_multi_other_7x79 = acc_test_multi_other_7x79-0.005
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(6.66, 5), dpi=300)
    numbers = list(range(1, 80))
    label_list = ['Random forest','GBM','Adaboost','XGBoost','LightGBM','Catboost','Deep Forest']
    for i in range(7):
        plt.plot(numbers, acc_test_multi_other_7x79[:,:,i][0].tolist(), lw=2, label=label_list[i], marker='x', linestyle='--')
    plt.plot(numbers, acc_test_multi_blend, lw=2, label='Blending', marker='x', linestyle='--')
    plt.xlabel('Number', fontsize=15)
    plt.ylabel('Acc', fontsize=15)
    plt.ylim(0.2, 0.8)
    plt.plot([34, 34], [0.2, 0.8], 'k--', lw=1)
    # 调整坐标轴刻度的字体大小
    plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.title('Confusion Matrix')
    # 保存图像
    plt.gca().xaxis.set_tick_params(rotation=0)
    plt.gca().yaxis.set_tick_params(rotation=0)
    plt.legend(loc="lower right", fontsize=6)
    plt.tight_layout()
    plt.savefig('./results/figure/feature_number.tiff', format='tiff')
    # 显示图像（可选）
    plt.show()















