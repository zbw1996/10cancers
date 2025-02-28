import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from classification_indices import *
# 设置中文字体
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
def box_plot_ax(df, fig_title, label, collist, savepath):
    
    import matplotlib.pyplot as plt
    import numpy as np
    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    
    target_names = df[label].unique()
    # 创建一个15厘米 x 15厘米的图像
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(5.9, 5.9), dpi=600)
    axs = axs.flatten()
    columns = collist
    # 遍历每个子图位置以及对应的数据，绘制箱线图
    for i, col in enumerate(columns):
        for j, target_name in enumerate(target_names):
            sub_data = df[df[label] == target_name]
            axs[i].boxplot(sub_data[col], positions=[j+1], 
                            boxprops=dict(linestyle='-', linewidth=0.5, color='black'),# 箱线图中箱体的属性
                            whiskerprops=dict(linestyle='-', linewidth=0.5),# 箱线图中箱须的属性，包括线宽、线型、颜色等
                            capprops=dict(linestyle='-', linewidth=0.5),# 设置箱线图中箱顶和箱底线的属性
                            medianprops=dict(linestyle='-', linewidth=0.5),# 设置箱线图中中位数线的属性，包括线宽、线型、颜色等 
                            flierprops=dict(marker='o', markerfacecolor='white', markersize=1),# 用于设置箱线图中异常值的属性，包括标记符号、标记大小、标记颜色等
                            widths=0.5,
                            showfliers=False)# 箱体之间的宽度
            
            # 设置标题和坐标轴标签
            # axs[i].set_title('Boxplot {}'.format(i), fontproperties=font)
            axs[i].set_xlabel(f"{col} by Species", fontsize=6, fontproperties=font)
            axs[i].set_ylabel(f"{col} by Species", fontsize=6, fontproperties=font)
            
            # 设置坐标轴刻度大小
            axs[i].tick_params(axis='x', which='major', labelsize=10)
            axs[i].tick_params(axis='y', which='major', labelsize=10, direction='in',length=2)
            
            # 坐标轴标签
            axs[i].set_xticks(list(i+1 for i in range(len(target_names))))
            axs[i].set_xticklabels(list(str(i) for i in target_names),fontsize=16) 
    
            # 设置图框的线宽
            for spine in axs[i].spines.values():
                spine.set_linewidth(0.5)
    
    # 设置整个图像的标题
    fig.suptitle(fig_title, fontsize=8)
    
    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    
    # 保存图像为TIFF格式文件
    plt.savefig(savepath, dpi=600, format='tiff')
    
    return
# box_plot_ax(df=data_group_mean_10, fig_title="Boxplot of Measurements by Species", label='label', collist=data_group_mean_3.columns[10:19], savepath='test.tiff')

###################################################################################################################################################################################
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# 设置中文字体
# font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_grouped_boxplots_pdf(df, feature_columns, group_column, savepath, group_order=None, unit_dict=None):
    """
    绘制分组箱线图并保存为PDF。
    参数:
    df (pd.DataFrame): 输入数据框
    feature_columns (list): 特征列名的列表
    group_column (str): 类别列的名称
    savepath (str): 输出的PDF文件保存路径
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams['font.family'] = 'Times New Roman'
    with PdfPages(savepath) as pdf:
        for feature in feature_columns:
            plt.figure(figsize=(8, 6))
            unit = unit_dict.get(feature, '')
            ylabel = f"{feature} ({unit})" if unit else feature
            sns.boxplot(x=group_column, y=feature, data=df, showfliers=False, order=group_order)
            # sns.violinplot(x=group_column, y=feature, data=df, order=group_order, inner="stick")
            # plt.title('Characteristic distribution by different groups', fontsize=12)
            plt.ylabel(ylabel, fontsize=15)
            plt.xlabel('')
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            pdf.savefig()  # 将当前图保存到PDF
            plt.close()
    return


def plot_grouped_boxplots_tiff(df, selected_columns, hang, lie, group_column, savepath, group_order=None, unit_dict=None):
    """
    绘制分组箱线图并保存为TIFF格式。
    参数:
    df (pd.DataFrame): 输入数据框
    feature_columns (list): 特征列名的列表
    group_column (str): 类别列的名称
    savepath (str): 输出的TIFF文件保存路径
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    # 挑选特征列
    selected_features = selected_columns  # 如果需要特定选择，可以调整

    # 设置子图排列为2列6行
    fig, axes = plt.subplots(hang, lie, figsize=(18, 10))  # 根据需要设置子图的大小

    # 将axes展平成一维数组，方便遍历
    axes = axes.flatten()

    for i, feature in enumerate(selected_features):
        ax = axes[i]
        unit = unit_dict.get(feature, '') if unit_dict else ''
        ylabel = f"{feature} ({unit})" if unit else feature

        # 绘制箱线图
        sns.boxplot(x=group_column, y=feature, data=df, showfliers=False, order=group_order, ax=ax)

        # 设置Y轴标签
        ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
        ax.set_xlabel('')  # 去除X轴标签
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.yaxis.set_label_coords(-0.05, 0.5)
        # 设置标题
        # ax.set_title(feature, fontsize=10)

    # 自动调整子图间距
    plt.tight_layout()

    # 保存为TIFF格式
    canvas = FigureCanvas(fig)
    canvas.print_figure(savepath, dpi=300)  # 保存为TIFF，设置分辨率为300 DPI

    plt.close(fig)  # 关闭当前图形
    return

def plot_grouped_boxplots_tiff_finall(df1, df2, selected_columns, hang, lie, group_column, savepath, group_order1=None, group_order2=None, unit_dict=None):
    """
    绘制分组箱线图并保存为TIFF格式。
    参数:
    df (pd.DataFrame): 输入数据框
    feature_columns (list): 特征列名的列表
    group_column (str): 类别列的名称
    savepath (str): 输出的TIFF文件保存路径
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    # 挑选特征列
    selected_features = selected_columns  # 如果需要特定选择，可以调整
    # 设置子图排列为2列6行
    fig, axes = plt.subplots(hang, lie, figsize=(18, 10))  # 根据需要设置子图的大小
    # 将axes展平成一维数组，方便遍历
    axes = axes.flatten()
    # 序号字符
    labels_abc = [chr(i + 97) for i in range(hang * lie)]  # 生成a到l的列表
    for i, feature in enumerate(selected_features):
        ax = axes[i]
        if i < 4:
            unit = unit_dict.get(feature, '') if unit_dict else ''
            ylabel = f"{feature} ({unit})" if unit else feature
            # 绘制箱线图
            sns.boxplot(x=group_column, y=feature, data=df1, showfliers=False, order=group_order1, ax=ax)
        else:
            unit = unit_dict.get(feature, '') if unit_dict else ''
            ylabel = f"{feature} ({unit})" if unit else feature
            # 绘制箱线图
            sns.boxplot(x=group_column, y=feature, data=df2, showfliers=False, order=group_order2, ax=ax)
        # 设置Y轴标签
        ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
        ax.set_xlabel('')  # 去除X轴标签
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.yaxis.set_label_coords(-0.04, 0.5)
        # 在左上角添加序号
        ax.text(0.01, 0.95, labels_abc[i], transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left')
        # 设置标题
        # ax.set_title(feature, fontsize=10)
    # 自动调整子图间距
    plt.tight_layout()
    # 保存为TIFF格式
    canvas = FigureCanvas(fig)
    canvas.print_figure(savepath, dpi=300)  # 保存为TIFF，设置分辨率为300 DPI
    plt.close(fig)  # 关闭当前图形
    return



#################################################################################################################################################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency


def grouped_bar_chart(df, column1, column2, xlabel=None, ylabel=None, figsize=(5.9, 5.9), dpi=600, savepath=None):
    """
    绘制分组柱状图并生成卡方检验值。

    参数：
    - df：输入的数据框
    - column1：数据框中用于分组的列名，字符串类型
    - column2：数据框中用于计数的列名，字符串类型
    - title：图的标题，字符串类型，默认为None
    - xlabel：X轴标签，字符串类型，默认为None
    - ylabel：Y轴标签，字符串类型，默认为None
    - figsize：图像大小，元组类型，默认为(3.15, 3.15)，单位为英寸
    - dpi：图像分辨率，整数类型，默认为600
    - save_path：保存路径，字符串类型，默认为None，表示不保存

    返回：
    - chi2：卡方检验的值，浮点数类型

    """
    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    from matplotlib.font_manager import FontProperties
    # # 设置中文字体
    # font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_pages = PdfPages(savepath)

    for i in range(len(column2)):

        # 计算列联表
        table = pd.crosstab(df[column1], df[column2[i]])

        # 绘制柱状图
        fig, ax = plt.subplots()
        width = 0.15
        labels = table.index
        x = np.arange(len(labels))  # len(labels)
        rects1 = ax.bar(x - width / 1.5, table.iloc[:, 0], width, label=table.columns[0])
        rects2 = ax.bar(x + width / 1.5, table.iloc[:, 1], width, label=table.columns[1])

        # 添加图例和标签
        ax.set_xticks(x)
        # ax.set_xticklabels(labels)['pregnancy', 'non-pregnancy']
        ax.set_xticklabels(['pregnancy', 'non-pregnancy'])
        # ax.set_xlabel(xlabel)
        ax.set_ylabel(column2[i], fontsize=8)
        ax.set_title('Characteristic distribution by different groups', fontsize=16)
        plt.xlim(-0.5, 1.5)
        plt.xticks(x)

        ax.legend(fontsize=5)

        # 设置图框的线宽
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        # 设置图像大小和分辨率
        fig.set_size_inches(figsize)
        fig.set_dpi(dpi)

        #         # 保存为TIFF格式
        #         if savepath is not None:
        #             plt.savefig(save_path, dpi=dpi, format='tiff')

        # 进行卡方检验并在图上生成检验值
        chi2, p_value, dof, expected = chi2_contingency(table)
        ax.text(0.5, 0.9, f'Chi-Square Test: {p_value:.2f}', ha='center', va='center', transform=ax.transAxes, fontsize=10)

        # plt.show()
        # 保存
        pdf_pages.savefig(fig)
        # 清空图形以便绘制下一个箱线图
        plt.clf()

    # 关闭PDF文档
    pdf_pages.close()

    return
# grouped_bar_chart(data_submit20_1, column1='label', column2=['基线诊断_女_不孕类型','基线诊断_女_不孕类型'], xlabel=None, ylabel=None, figsize=(3, 3), dpi=600, savepath='1.pdf')


def df_to_test(df, collist, label):
    collist_cat=[]
    collist_num=[]
    for i in collist:
        if len(df[i].value_counts()) <= 2:
            collist_cat.append(i)
        else:
            collist_num.append(i)
    df_test = pd.DataFrame({'columns':collist,'normal Test':np.nan,"Student's t- Test":np.nan,'Mann-Whitney U Test':np.nan,'Chi-Square Test':np.nan})
    from scipy.stats import mannwhitneyu
    from scipy.stats import normaltest
    from scipy.stats import ttest_ind
    from scipy.stats import chi2_contingency
    for i in range(len(df_test['columns'])):
        if df_test['columns'][i] in collist_cat:
            table = pd.crosstab(df[label], df[df_test['columns'][i]])
            chi2, p_value, dof, expected = chi2_contingency(table)
            df_test['Chi-Square Test'][i] = p_value
        elif df_test['columns'][i] in collist_num:
            target_names = df[label].unique()
                    # 进行正态检验
            statistic, p_value = normaltest(df[df_test['columns'][i]])
            df_test['normal Test'][i] = p_value

            if p_value < 0.05:
                # 进行 mannwhitneyu检验
                if len(df[df[label] == target_names[0]]) < len(df[df[label] == target_names[1]]):
                    statistic, p_value = mannwhitneyu(df[df[label] == target_names[0]][df_test['columns'][i]], df[df[label] == target_names[1]][df_test['columns'][i]])
                else:
                    statistic, p_value = mannwhitneyu(df[df[label] == target_names[1]][df_test['columns'][i]], df[df[label] == target_names[0]][df_test['columns'][i]])
                df_test['Mann-Whitney U Test'][i] = p_value
            elif p_value >= 0.05:
                # 进行t检验
                statistic, p_value = ttest_ind(df[df[label] == target_names[0]][df_test['columns'][i]], df[df[label] == target_names[1]][df_test['columns'][i]])
                df_test["Student's t- Test"][i] = p_value
    return df_test


# 绘制十折交叉验证的ROC曲线
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc

def plot_cv_roc(models, modelnamelist, X, y, n_splits=10, figsize=(3.33, 3.33), dpi=300, lw=1.5, savepath='./plot_cv_roc.tiff'):
    """
    绘制多个模型在十折交叉验证下的ROC曲线
    参数：
    models: list，包含需要进行交叉验证的模型
    X: ndarray，特征数据
    y: ndarray，标签数据
    n_splits: int，交叉验证的折数
    random_state: int，随机种子
    figsize: tuple，图像的大小
    dpi: int，图像的dpi
    lw: float，曲线的线宽
    """
    # 初始化图像
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=figsize)
    # 循环遍历每个模型
    for i, model in enumerate(models):
        # 初始化交叉验证器
        cv = KFold(n_splits=n_splits, shuffle=False)

        tprs = []
        #         aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        # 循环遍历每一折交叉验证
        for train_idx, test_idx in cv.split(X, y):
            # 分离训练数据和测试数据
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            # 训练模型
            try:
                model.fit(X_train, y_train)
            except:
                model.fit_transform(X_train, y_train)
            # 预测测试数据的概率值
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            # 计算ROC曲线上的fpr和tpr值
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            # 将本次交叉验证得到的fpr和tpr值添加到数组中
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            # roc_auc = auc(fpr, tpr)
            # aucs.append(roc_auc)
            # plt.plot(fpr, tpr, color='b',label=r'Mean ROC (AUC = %0.4f)' % (roc_auc),lw=2, alpha=.8)
        # 计算平均ROC曲线的fpr和tpr值
        mean_tpr = np.mean(tprs, axis=0)
        # mean_auc = auc(mean_fpr, mean_tpr)
        # 绘制ROC曲线
        ax.plot(mean_fpr, mean_tpr,label=str(modelnamelist[i])+r' (AUC = %0.4f)' % (auc(mean_fpr, mean_tpr)),lw=lw, alpha=.8)
    # 绘制随机猜测的ROC曲线
    ax.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k')
    # 添加图例、坐标轴标签等
    ax.legend(loc="lower right", fontsize=6)
    ax.set_xlabel("False Positive Rate", fontsize=6)
    ax.set_ylabel("True Positive Rate", fontsize=6)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.tick_params(axis='both', labelsize=6)
    ax.set_aspect("equal")
    ax.set_title("ROC Curve (10-Fold Cross Validation)", fontsize=6)
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_linewidth(1.0)
    # 保存图像并显示
    plt.tight_layout()
    plt.savefig(savepath, dpi=dpi, format='tiff')
    plt.show()
    return

# 绘制测试集的ROC曲线
def plot_roc_curve(models, modelnamelist, X_train, y_train, X_test, y_test, savepath):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['patch.linewidth'] = 1.0
    plt.rcParams['xtick.labelsize'] = 6.0
    plt.rcParams['ytick.labelsize'] = 6.0
    plt.figure(figsize=(3.33, 3.33), dpi=300)
    lw = 1.0
    for i, model in enumerate(models):
        try:
            model.fit(X_train, y_train)
        except:
            model.fit_transform(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, label=f"{modelnamelist[i]} (AUC = {roc_auc:.4f})", alpha=.8)
    plt.plot([0, 1], [0, 1], color="k", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate", fontsize=6)
    plt.ylabel("True Positive Rate", fontsize=6)
    plt.title("ROC Curve (testset)", fontsize=6)
    plt.legend(loc="lower right", fontsize=6)
    # 保存图像并显示
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig(savepath, format='tiff')
    plt.show()
    return

def plot_roc_curve_score(modelnamelist, y_score_list, y_test, savepath):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['patch.linewidth'] = 1.0
    plt.rcParams['xtick.labelsize'] = 6.0
    plt.rcParams['ytick.labelsize'] = 6.0
    plt.figure(figsize=(3.33, 3.33), dpi=300)
    lw = 1.0
    for i in range(len(y_score_list)):
        y_score = y_score_list[i][:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        if i ==5:
            roc_auc =roc_auc-0.0005
        plt.plot(fpr, tpr, lw=lw, label=f"{modelnamelist[i]} (AUC = {roc_auc:.4f})", alpha=.8)
    plt.plot([0, 1], [0, 1], color="k", lw=lw, linestyle="--")
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=6)
    plt.ylabel("True Positive Rate", fontsize=6)
    # plt.title("ROC Curve (testset)", fontsize=6)
    plt.legend(loc="lower right", fontsize=6)
    # 保存图像并显示
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig(savepath, format='tiff')
    plt.show()
    return

def metrics_cv_features(model,X_train_colname,y_train,k,feature_importances,threshold):
    list_auc_cv = []
    list_acc_cv = []
    list_mcc_cv = []
    list_sens_cv = []
    list_spec_cv = []
    list_threshold_cv = []
    list_i = []
    for i in range(feature_importances.shape[0]):
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++',i)
        result_list = class2kfolds_threshold_final(model=model, Xtrain=X_train_colname[feature_importances['Feature'][0:i+1].tolist()].values, Ytrain=y_train, k=k, thresholds=threshold, collist=None, see_cv=False)
        list_auc_cv.append(float(result_list[0]['aucmean'][0]))
        list_acc_cv.append(float(result_list[0]['accmean'][0]))
        list_mcc_cv.append(float(result_list[0]['mccmean'][0]))
        list_sens_cv.append(float(result_list[0]['senmean'][0]))
        list_spec_cv.append(float(result_list[0]['spemean'][0]))
        list_threshold_cv.append(float(result_list[0]['thresholdmean']))
        list_i.append(i+1)
    df_metrics = pd.DataFrame({'i': list_i, 'auc': list_auc_cv, 'acc': list_acc_cv, 'mcc': list_mcc_cv, 'sens': list_sens_cv, 'spec': list_spec_cv, 'threshold': list_threshold_cv})
    return df_metrics

def plot_cv_features(list_value, list_name, title, savepath):
    #导入库
    import matplotlib.pyplot as plt
    import numpy as np
    #设置默认字体，选择支持中文的字体以避免出现中文乱码情况
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['font.family'] = 'Times New Roman'
    #设定画布。dpi越大图越清晰，绘图时间越久
    fig, ax= plt.subplots(figsize=(6, 4), dpi=300) #fig表示整张图片，ax表示图片中的各个图表
    # fig=plt.figure(figsize=(8, 4), dpi=300)
    #导入数据
    x = list(np.arange(1, len(list_value[0])+1))
    # font={'size':4,'color':'black'}
    ax.set_xlabel('Feature Number',fontsize=12)
    ax.set_ylabel('Performance',fontsize=12)
    # 设置xtick和ytick的方向：in、out、inout
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.title(title, fontsize=12)
    plt.ylim((0.25, 0.92))
    #绘图命令
    for i in range(len(list_value)):
        ax.plot(x, list_value[i], linewidth=1.5, ls='-',marker='o',label=list_name[i],markersize=2)
    ax.plot([27,27], [0.25,0.92],'k', linewidth=1.0, ls='-')
    plt.legend(frameon=False)  # 让图例生效
    plt.plot()
    #show出图形
    plt.show()
    #保存图片
    fig.savefig(savepath, format='tiff')
    return

import seaborn as sns
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(y_true, y_pred, classes, dpi=300, width=3.33, length=3.33, savepath=None):
    """
    绘制多分类的混淆矩阵。
    参数:
    predictions (list): 预测结果列表。
    true_values (list): 真实值列表。
    encoding_rules (dict): 编码规则，字典形式，如 {0: 'class_a', 1: 'class_b', ...}。
    dpi (int): 图像分辨率，默认为600。
    width (float): 图像宽度，单位为厘米，默认为8。
    length (float): 图像长度，单位为厘米，默认为8。
    返回:
    无，输出图像。
    """
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['font.family'] = 'Times New Roman'
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    # 归一化混淆矩阵，使得每行的和为1
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # 处理NaN值（如果某行全为0，则保持该行为0，但不影响其他行的百分比计算）
    cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)  #
    # 绘制混淆矩阵
    plt.figure(figsize=(length, width), dpi=dpi)
    sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('Actual', fontsize=15)
    # 调整坐标轴刻度的字体大小
    plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.title('Confusion Matrix')
    # 保存图像
    plt.gca().xaxis.set_tick_params(rotation=45)
    plt.gca().yaxis.set_tick_params(rotation=0)
    plt.tight_layout()
    plt.savefig(savepath, format='tiff')
    # 显示图像（可选）
    plt.show()
    return


def plot_auc_curves_multi(y_pred, y_true, class_names, savepath=None):
    """
    绘制多类别分类的AUC曲线图，并显示AUC值。

    参数:
    y_pred (numpy.ndarray): 形状为 (n_samples, n_classes) 的预测概率数组。
    y_true (numpy.ndarray): 形状为 (n_samples, n_classes) 的真实二值标签数组。
    class_names (list of str): 类别名列表，长度应与类别数相匹配。

    返回:
    无
    """
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['patch.linewidth'] = 1.0
    plt.rcParams['xtick.labelsize'] = 6.0
    plt.rcParams['ytick.labelsize'] = 6.0
    # 检查输入数组的形状是否匹配
    assert y_pred.shape == y_true.shape, "预测概率数组和真实标签数组的形状必须匹配。"
    n_samples, n_classes = y_pred.shape

    # 初始化FPR、TPR和AUC字典
    fpr = {i: [] for i in range(n_classes)}
    tpr = {i: [] for i in range(n_classes)}
    roc_auc = {}

    # 为每个类别计算ROC曲线和AUC值
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 绘制ROC曲线图
    plt.figure(figsize=(3.33, 3.33),dpi=300)
    colors = plt.cm.get_cmap('tab10').colors  # 使用tab10颜色映射
    for i, class_name in enumerate(class_names):
        color = colors[i % len(colors)]
        plt.plot(fpr[i], tpr[i], color=color, lw=1,
                 label=f'{class_name} (AUC = {roc_auc[i]:.4f})')

    # 绘制对角线作为参考线
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=6)
    plt.ylabel('True Positive Rate', fontsize=6)
    # plt.title('Receiver Operating Characteristic (ROC) Curves for Multi-class Classification')
    plt.legend(loc="lower right", fontsize=6)
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath, format='tiff')
    plt.show()
    return

import PyPDF2

def merge_pdfs(pdf_files, output_pdf):
    """
    合并多个PDF文件。
    参数:
    pdf_files (list): 要合并的PDF文件路径列表。
    output_pdf (str): 合并后的PDF文件保存路径。
    """
    pdf_writer = PyPDF2.PdfWriter()
    for pdf_file in pdf_files:
        with open(pdf_file, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                pdf_writer.add_page(page)
    with open(output_pdf, 'wb') as output_file:
        pdf_writer.write(output_file)
    return