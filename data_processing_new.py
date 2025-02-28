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
from data_preprocess import *
from plot_figurer import *
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


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

if __name__ == '__main__':
    # # parameter
    # row_null_drop = 0.2  # 行缺失超过0.3删除
    # col_null_drop = 0.2  # 列缺失超过0.3删除
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
    # data_count_encode_clean = pd.read_csv('./dataset/Supplementary_encode_data.csv', encoding='gbk')
    #
    # # # 3.特征工程
    # data_process = data_count_encode_clean.copy()
    # for i in data_process.columns:
    #     try:
    #         if data_count_encode_clean[i].std() == 0:
    #             print(i)
    #             del data_process[i]
    #     except:
    #         pass
    # # data_process2 = data_process.loc[:, data_process.isnull().mean(axis=0) < 0.9]
    #
    # # # 3.数据集划分
    # listX1, listX2, listX3, listX4, listX5 = columns_statistics(data_process)
    # data_n = data_process.loc[data_process['Diagnose'] == 'Normal', list_add([listnomodel, ['Sex', 'Age'], listX1, listX2, listX3])]
    # data_o = data_process.loc[data_process['Diagnose'] == 'Other disease', list_add([listnomodel, ['Sex', 'Age'], listX1, listX2, listX3])]
    # data_cancer = data_process[(data_process['Diagnose'] != 'Normal') & (data_process['Diagnose'] != 'Other disease')]
    # data_cancer_model1 = data_cancer[list_add([listnomodel, ['Sex', 'Age'], listX1, listX2, listX3])]
    #
    # data_cancer_model1_drop1, dellist_cancer_model1_drop1 = df_drop_row_col(df=data_cancer_model1,
    #                                                                         collist=list_minus(list(data_cancer_model1.columns), listnomodel),
    #                                                                         row=row_null_drop,
    #                                                                         col=col_null_drop, k=100)
    # data_cancer_model2_drop1, dellist_cancer_model2_drop1 = df_drop_row_col(df=data_cancer,
    #                                                                         collist=list_minus(list(data_cancer.columns), listnomodel),
    #                                                                         row=row_null_drop,
    #                                                                         col=col_null_drop, k=100)
    #
    #
    #
    # data_n_drop1 = data_n.loc[:, data_n.isnull().mean(axis=0) < 0.9]
    # data_o_drop1 = data_n.loc[:, data_o.isnull().mean(axis=0) < 0.9]
    # df_test = data_process.loc[:,intersect_lists([data_cancer_model1_drop1.columns, data_n_drop1.columns, data_o_drop1.columns])]
    # df_test_drop1, dellist_df_test_drop1 = df_drop_row_col(df=df_test,
    #                                                         collist=list_minus(list(df_test.columns), listnomodel),
    #                                                         row=row_null_drop,
    #                                                         col=col_null_drop, k=100)
    # df_test_drop2, dellist_df_test_drop2 = df_drop_row_col(df=data_process.loc[:,list_add([listnomodel, ['Sex', 'Age'], listX1, listX2, listX3, listX4, listX5])],
    #                                                         collist=list_add([['Sex', 'Age'], listX1, listX2, listX3]),
    #                                                         row=row_null_drop,
    #                                                         col=col_null_drop, k=100)
    # df_test_drop3, dellist_df_test_drop3 = df_drop_row_col(df=data_process.loc[data_process['Diagnose'] != 'Normal',list_add([listnomodel, ['Sex', 'Age'], listX1, listX2, listX3])],
    #                                                         collist=list_add([['Sex', 'Age'], listX1, listX2, listX3]),
    #                                                         row=row_null_drop,
    #                                                         col=col_null_drop, k=100)
    # df_test_drop4, dellist_df_test_drop4 = df_drop_row_col(df=data_process.loc[:, list_add([listnomodel, ['Sex', 'Age'], listX1])],
    #                                                         collist=list_add([['Sex', 'Age'], listX1]),
    #                                                         row=row_null_drop,
    #                                                         col=col_null_drop, k=100)
    # df_test_drop5, dellist_df_test_drop5 = df_drop_row_col(df=data_process.loc[:, list_add([listnomodel, ['Sex', 'Age'], listX2])],
    #                                                         collist=list_add([['Sex', 'Age'], listX2]),
    #                                                         row=row_null_drop,
    #                                                         col=col_null_drop, k=100)
    # df_test_drop6, dellist_df_test_drop6 = df_drop_row_col(df=data_process.loc[:, list_add([listnomodel, ['Sex', 'Age'], listX3])],
    #                                                         collist=list_add([['Sex', 'Age'], listX3]),
    #                                                         row=row_null_drop,
    #                                                         col=col_null_drop, k=100)
    # df_test_drop7, dellist_df_test_drop7 = df_drop_row_col(df=data_process.loc[:, list_add([listnomodel, ['Sex', 'Age'], listX1, listX2])],
    #                                                         collist=list_add([['Sex', 'Age'], listX1, listX2]),
    #                                                         row=row_null_drop,
    #                                                         col=col_null_drop, k=100)
    # df_test_drop8, dellist_df_test_drop8 = df_drop_row_col(df=data_process.loc[:, list_add([listnomodel, ['Sex', 'Age'], listX2, listX3])],
    #                                                         collist=list_add([['Sex', 'Age'], listX2, listX3]),
    #                                                         row=row_null_drop,
    #                                                         col=col_null_drop, k=100)
    # # def test_model(df):
    # #     df_fill = df_fillna(df=df, method1='median', numcollist=list_minus(df.columns, listnomodel))  # 填充缺失值
    # #     X = df_fill.loc[:, list_minus(df.columns, listnomodel)]
    # #     Y = df_fill.loc[:, 'Cancer']
    # #     from sklearn.model_selection import train_test_split
    # #     X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, test_size=0.2, random_state=100)
    # #     from sklearn.ensemble import RandomForestClassifier as RFC
    # #     clf = RFC(n_estimators=1000, random_state=42)
    # #     clf.fit(X_train, y_train)
    # #     y_pred = clf.predict(X_test)
    # #     y_pred_proba = clf.predict_proba(X_test)[:,1]
    # #     from sklearn.metrics import roc_auc_score
    # #     from sklearn.metrics import accuracy_score
    # #     accuracy = accuracy_score(y_test, y_pred)
    # #     print(f'Accuracy: {accuracy:.2f}')
    # #     auc = roc_auc_score(y_test, y_pred_proba)
    # #     print(f'AUC: {auc:.2f}')
    # #     return
    #
    # # 填充X1X2X3但不填充X4X5
    # df_test_drop2_fill = df_fillna(df=df_test_drop2, method1='median', numcollist=list_minus(df_test_drop2.columns, list_add([listnomodel, listX4, listX5])))
    # # 对尿液颜色进行独热编码
    # df_test_drop2_fill_onehot, onehot_encoder, kmeans_model = preprocess_data_original(df_test_drop2_fill, num_col=None, cat_col=['X301'], non_cat_col=None)
    #
    # # # 低方差过滤
    # var_filter_k = 0
    # df_var_filter = var_filter(df_test_drop2_fill_onehot, var_filter_k, list_minus(df_test_drop2_fill_onehot.columns, list_add([listnomodel, listX4, listX5])))
    # # pearson_corr过滤
    # pearson_corr_k = 0.99
    # df_pearson_corr = pearson_corr(df_var_filter, pearson_corr_k, list_minus(df_var_filter.columns, list_add([listnomodel, listX4, listX5])))
    #
    # from sklearn.model_selection import train_test_split
    #
    # X_train, X_vt, y_train, y_vt = train_test_split(df_pearson_corr, df_pearson_corr['Cancer'], test_size=0.40, random_state=15)  # 指定特征值占1/5,随机数种子是15
    # X_valid, X_test, y_valid, y_test = train_test_split(X_vt, y_vt, test_size=0.50, random_state=15)  # 指定特征值占1/5,随机数种子是15
    # pd.DataFrame(X_train['Diagnose'].value_counts())
    # pd.DataFrame(X_valid['Diagnose'].value_counts())
    # pd.DataFrame(X_test['Diagnose'].value_counts())
    #
    # table1_all = calculate_statistics_by_category(df=df_test_drop2_fill, category_column='Diagnose',
    #                                                           gender_column='Sex', age_column='Age')
    # table1_train = calculate_statistics_by_category(df=X_train, category_column='Diagnose',
    #                                                           gender_column='Sex', age_column='Age').loc[:, ['Category', 'Female Count', 'Male Count']]
    # table1_valid = calculate_statistics_by_category(df=X_valid, category_column='Diagnose',
    #                                                           gender_column='Sex', age_column='Age').loc[:, ['Category', 'Female Count', 'Male Count']]
    # table1_test = calculate_statistics_by_category(df=X_test, category_column='Diagnose',
    #                                                           gender_column='Sex', age_column='Age').loc[:, ['Category', 'Female Count', 'Male Count']]
    # for i in [table1_train, table1_valid, table1_test]:
    #     table1_all = pd.merge(table1_all,i, on='Category', sort=False)
    # table1_all.to_csv('./results/table/table_1.csv', index=False)
    # X_train['dataset'] = 'Cross-validation set'
    # X_valid['dataset'] = 'Validation set'
    # X_test['dataset'] = 'Test set'
    # Supplementary_process_data = pd.concat([X_train, X_valid, X_test], ignore_index=True)
    # Supplementary_process_data.to_csv('./dataset/Supplementary_process_data.csv', encoding='gbk', index=False, na_rep='NA')


    # # parameter
    row_null_drop = 0.2  # 行缺失超过0.3删除
    col_null_drop = 0.2  # 列缺失超过0.3删除
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

    # # 3.特征工程
    data_process = data_count_encode_clean.copy()
    for i in data_process.columns:
        try:
            if data_count_encode_clean[i].std() == 0:
                print(i)
                del data_process[i]
        except:
            pass
    # data_process2 = data_process.loc[:, data_process.isnull().mean(axis=0) < 0.9]

    # # 3.数据集划分
    listX1, listX2, listX3, listX4, listX5 = columns_statistics(data_process)
    data_n = data_process.loc[data_process['Diagnose'] == 'Normal', list_add([listnomodel, ['Sex', 'Age'], listX1, listX2, listX3])]
    data_o = data_process.loc[data_process['Diagnose'] == 'Other disease', list_add([listnomodel, ['Sex', 'Age'], listX1, listX2, listX3])]
    data_cancer = data_process[(data_process['Diagnose'] != 'Normal') & (data_process['Diagnose'] != 'Other disease')]
    data_cancer_model1 = data_cancer[list_add([listnomodel, ['Sex', 'Age'], listX1, listX2, listX3])]

    data_cancer_model1_drop1, dellist_cancer_model1_drop1 = df_drop_row_col(df=data_cancer_model1,
                                                                            collist=list_minus(list(data_cancer_model1.columns), listnomodel),
                                                                            row=row_null_drop,
                                                                            col=col_null_drop, k=100)
    data_cancer_model2_drop1, dellist_cancer_model2_drop1 = df_drop_row_col(df=data_cancer,
                                                                            collist=list_minus(list(data_cancer.columns), listnomodel),
                                                                            row=row_null_drop,
                                                                            col=col_null_drop, k=100)



    data_n_drop1 = data_n.loc[:, data_n.isnull().mean(axis=0) < 0.9]
    data_o_drop1 = data_n.loc[:, data_o.isnull().mean(axis=0) < 0.9]
    df_test = data_process.loc[:,intersect_lists([data_cancer_model1_drop1.columns, data_n_drop1.columns, data_o_drop1.columns])]
    df_test_drop1, dellist_df_test_drop1 = df_drop_row_col(df=df_test,
                                                            collist=list_minus(list(df_test.columns), listnomodel),
                                                            row=row_null_drop,
                                                            col=col_null_drop, k=100)
    df_test_drop2, dellist_df_test_drop2 = df_drop_row_col(df=data_process.loc[:,list_add([listnomodel, ['Sex', 'Age'], listX1, listX2, listX3, listX4, listX5])],
                                                            collist=list_add([['Sex', 'Age'], listX1, listX2, listX3]),
                                                            row=row_null_drop,
                                                            col=col_null_drop, k=100)
    df_test_drop3, dellist_df_test_drop3 = df_drop_row_col(df=data_process.loc[data_process['Diagnose'] != 'normal',list_add([listnomodel, ['Sex', 'Age'], listX1, listX2, listX3])],
                                                            collist=list_add([['Sex', 'Age'], listX1, listX2, listX3]),
                                                            row=row_null_drop,
                                                            col=col_null_drop, k=100)
    df_test_drop4, dellist_df_test_drop4 = df_drop_row_col(df=data_process.loc[:, list_add([listnomodel, ['Sex', 'Age'], listX1])],
                                                            collist=list_add([['Sex', 'Age'], listX1]),
                                                            row=row_null_drop,
                                                            col=col_null_drop, k=100)
    df_test_drop5, dellist_df_test_drop5 = df_drop_row_col(df=data_process.loc[:, list_add([listnomodel, ['Sex', 'Age'], listX2])],
                                                            collist=list_add([['Sex', 'Age'], listX2]),
                                                            row=row_null_drop,
                                                            col=col_null_drop, k=100)
    df_test_drop6, dellist_df_test_drop6 = df_drop_row_col(df=data_process.loc[:, list_add([listnomodel, ['Sex', 'Age'], listX3])],
                                                            collist=list_add([['Sex', 'Age'], listX3]),
                                                            row=row_null_drop,
                                                            col=col_null_drop, k=100)
    df_test_drop7, dellist_df_test_drop7 = df_drop_row_col(df=data_process.loc[:, list_add([listnomodel, ['Sex', 'Age'], listX1, listX2])],
                                                            collist=list_add([['Sex', 'Age'], listX1, listX2]),
                                                            row=row_null_drop,
                                                            col=col_null_drop, k=100)
    def test_model(df):
        df_fill = df_fillna(df=df, method1='median', numcollist=list_minus(df.columns, listnomodel))  # 填充缺失值
        X = df_fill.loc[:, list_minus(df.columns, listnomodel)]
        Y = df_fill.loc[:, 'Cancer']
        print(X.shape)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, test_size=0.2, random_state=100)
        from sklearn.ensemble import RandomForestClassifier as RFC
        clf = RFC(n_estimators=1000, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:,1]
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy:.2f}')
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f'AUC: {auc:.2f}')
        return

    # 填充X1X2X3但不填充X4X5
    df_test_drop2_fill = df_fillna(df=df_test_drop2, method1='median', numcollist=list_minus(df_test_drop2.columns, list_add([listnomodel, listX4, listX5])))
    # 对尿液颜色进行独热编码
    df_test_drop2_fill_onehot, onehot_encoder, kmeans_model = preprocess_data_original(df_test_drop2_fill, num_col=None, cat_col=['X301'], non_cat_col=None)

    # # 低方差过滤
    var_filter_k = 0
    df_var_filter = var_filter(df_test_drop2_fill_onehot, var_filter_k, list_minus(df_test_drop2_fill_onehot.columns, list_add([listnomodel, listX4, listX5])))
    # pearson_corr过滤
    pearson_corr_k = 0.99
    df_pearson_corr = pearson_corr(df_var_filter, pearson_corr_k, list_minus(df_var_filter.columns, list_add([listnomodel, listX4, listX5])))

    from sklearn.model_selection import train_test_split

    X_train, X_vt, y_train, y_vt = train_test_split(df_pearson_corr, df_pearson_corr['Cancer'], test_size=0.40, random_state=15)  # 指定特征值占1/5,随机数种子是15
    X_valid, X_test, y_valid, y_test = train_test_split(X_vt, y_vt, test_size=0.50, random_state=15)  # 指定特征值占1/5,随机数种子是15
    pd.DataFrame(X_train['Diagnose'].value_counts())
    pd.DataFrame(X_valid['Diagnose'].value_counts())
    pd.DataFrame(X_test['Diagnose'].value_counts())

    table1_all = calculate_statistics_by_category(df=df_test_drop2_fill, category_column='Diagnose',
                                                              gender_column='Sex', age_column='Age')
    table1_train = calculate_statistics_by_category(df=X_train, category_column='Diagnose',
                                                              gender_column='Sex', age_column='Age').loc[:, ['Category', 'Female Count', 'Male Count']]
    table1_valid = calculate_statistics_by_category(df=X_valid, category_column='Diagnose',
                                                              gender_column='Sex', age_column='Age').loc[:, ['Category', 'Female Count', 'Male Count']]
    table1_test = calculate_statistics_by_category(df=X_test, category_column='Diagnose',
                                                              gender_column='Sex', age_column='Age').loc[:, ['Category', 'Female Count', 'Male Count']]
    for i in [table1_train, table1_valid, table1_test]:
        table1_all = pd.merge(table1_all,i, on='Category', sort=False)
    table1_all.to_csv('./results/table/table_1.csv', index=False)
    X_train['dataset'] = 'Cross-validation set'
    X_valid['dataset'] = 'Validation set'
    X_test['dataset'] = 'Test set'
    Supplementary_process_data = pd.concat([X_train, X_valid, X_test], ignore_index=True)
    # Supplementary_process_data.to_csv('./dataset/Supplementary_process_data.csv', encoding='gbk', index=False, na_rep='NA')











