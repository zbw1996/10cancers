import pandas as pd
import numpy as np
def data_expansion(dfdata):
    
    dfdata2 = dfdata.copy()
    
    for i in range(dfdata.shape[1]):
        for j in range(i):
            data = np.multiply(dfdata2.iloc[:,i].values.reshape(-1,1),dfdata2.iloc[:,j].values.reshape(-1,1))
            c={(str(dfdata.columns[i])+'*'+str(dfdata.columns[j])): list(data)}
            dfnew=pd.DataFrame(c)
            dfdata2 = pd.concat([dfdata2, dfnew],axis=1)
    
    for i in range(dfdata.shape[1]):
        for j in range(i):
            data = (dfdata2.iloc[:,i].values.reshape(-1,1))/(dfdata2.iloc[:,j].values.reshape(-1,1))
            c={(str(dfdata.columns[i])+'/'+str(dfdata.columns[j])): list(data)}
            dfnew=pd.DataFrame(c)
            dfdata2 = pd.concat([dfdata2, dfnew],axis=1)

    for i in range(dfdata.shape[1]):
        data = np.multiply(dfdata2.iloc[:,i].values.reshape(-1,1),dfdata2.iloc[:,i].values.reshape(-1,1))
        c = {(str(dfdata.columns[i]) + '**2'): list(data)}
        dfnew = pd.DataFrame(c)
        dfdata2 = pd.concat([dfdata2, dfnew], axis=1)
    
    for i in range(dfdata.shape[1],dfdata2.shape[1]):
        for j in range(dfdata2.shape[0]):
            dfdata2.iloc[j,i] = float(dfdata2.iloc[j,i])
    
    return dfdata2
    
    
def scale_np(X_train, X_test, method):
    """
    对训练集和测试集进行标准化
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, PowerTransformer, \
        QuantileTransformer, KernelCenterer
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'normalizer':
        scaler = Normalizer()
    elif method == 'power':
        scaler = PowerTransformer()
    elif method == 'quantile':
        scaler = QuantileTransformer()
    elif method == 'kernel':
        scaler = KernelCenterer()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

from sklearn import metrics
from sklearn.metrics import confusion_matrix # 混淆矩阵
import numpy as np
        
def sen(Y_test,Y_pred,n):
    #输入Y_test:0,1,2,3,4...
    sen = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)
        
    return sen

def pre(Y_test,Y_pred,n):
    
    pre = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fp = np.sum(con_mat[:,i]) - tp
        pre1 = tp / (tp + fp)
        pre.append(pre1)
        
    return pre

def spe(Y_test,Y_pred,n):
    spe = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)
    
    return spe

def ACC(Y_test,Y_pred,n):
    acc = []
    con_mat = confusion_matrix(Y_test,Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:,:])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i,:]) - tp
        fp = np.sum(con_mat[:,i]) - tp
        tn = number - tp - fn - fp
        acc1 = (tp + tn) / number
        acc.append(acc1)
        
    return acc
    
def ACCall(Y_test,Y_pred):
    acc = []
    tp = 0
    con_mat = confusion_matrix(Y_test,Y_pred)
    n = con_mat.shape[0]
    number = np.sum(con_mat[:, :])
    for i in range(n):
        tp = tp + con_mat[i][i]
    accall = tp/number
    
    return accall

# 从类生成实例
#sth1 = Something()
# 需要调用方法：
#result1 = sth1.func1(coef1, coef2, coef3)

from sklearn.model_selection import KFold
import pandas as pd
from sklearn.model_selection import GridSearchCV
# 二分类第一步选择模型进行网格搜索
# classifier = RFC()
# GS = GridSearchCV(classifier, param_grid = parameter, cv=10, score = "roc_auc")
# GS.fit(X_train, y_train)
# classifier_ = model
# classifier_.set_params(**GS.best_params_)
# classifier_输入一下二分类函数model = classifier_

def class2kfolds(model,Xtrain,Ytrain,k):
    kf = KFold(n_splits=k, shuffle=False)
    listauc = []
    listrecall = []
    listprecision = []
    listf1 = []
    listacc = []
    listsen = []
    listspe = []
    listmcc = []
    for i, (train_index, test_index) in enumerate(kf.split(Xtrain)):
        #print(f'KFold {i+1}:')
        #print("Train index:", train_index, "Test index:", test_index)
        X_train, X_test = Xtrain[train_index], Xtrain[test_index]
        y_train, y_test = Ytrain[train_index], Ytrain[test_index]
        clf = model.fit(X_train,y_train)
        y_pro = clf.predict_proba(X_test)[:,1]
        y_pred = clf.predict(X_test)
        auc = metrics.roc_auc_score(y_test, y_pro)
        recall = metrics.recall_score(y_test, y_pred, average='binary')
        precision = metrics.recall_score(y_test, y_pred, average='binary')
        f1 = metrics.f1_score(y_test, y_pred, average='binary')
        con_mat = confusion_matrix(y_test, y_pred, labels=[0,1])
        acc = (con_mat[0,0]+con_mat[1,1])/sum(sum(con_mat))
        sen = (con_mat[1,1]/sum(con_mat[1,:]))
        spe = (con_mat[0,0]/sum(con_mat[0,:]))
        mcc = (con_mat[0,0]*con_mat[1,1]-con_mat[0,1]*con_mat[1,0])/((sum(con_mat[1,:])*sum(con_mat[0,:])*sum(con_mat[:,0])*sum(con_mat[:,1]))**0.5)
        listauc.append(auc)
        listrecall.append(recall)
        listprecision.append(precision)
        listf1.append(f1)
        listacc.append(acc)
        listsen.append(sen)
        listspe.append(spe)
        listmcc.append(mcc)
    aucmean = sum(listauc)/len(listauc)
    recallmean = sum(listrecall)/len(listrecall)
    precisionmean = sum(listprecision)/len(listprecision)
    f1mean = sum(listf1)/len(listf1)
    accmean = sum(listacc)/len(listacc)
    senmean = sum(listsen)/len(listsen)
    spemean = sum(listspe)/len(listspe)
    mccmean = sum(listmcc)/len(listmcc)

    df = pd.DataFrame({'aucmean':('%.4f' % aucmean), 'recallmean':('%.4f' % recallmean), 'precisionmean':('%.4f' % precisionmean),
                       'f1mean':('%.4f' % f1mean) , 'accmean':('%.4f' % accmean), 'senmean':('%.4f' % senmean),
                       'spemean':('%.4f' % spemean), 'mccmean':('%.4f' % mccmean)},index=[0])

    return df

#例如model = SVC(kernel="rbf",C=2,gamma = 0.012742749857031322,cache_size=5000,probability=True)
#0，1分类
def class2kfolds_threshold(model,Xtrain,Ytrain,k,thresholds):
    kf = KFold(n_splits=k, shuffle=False)
    listauc = []
    listrecall = []
    listprecision = []
    listf1 = []
    listacc = []
    listsen = []
    listspe = []
    listmcc = []
    threshold = thresholds
    for i, (train_index, test_index) in enumerate(kf.split(Xtrain)):
        #print(f'KFold {i+1}:')
        #print("Train index:", train_index, "Test index:", test_index)
        X_train, X_test = Xtrain[train_index], Xtrain[test_index]
        y_train, y_test = Ytrain[train_index], Ytrain[test_index]
        clf = model.fit(X_train,y_train)
        y_pro = clf.predict_proba(X_test)[:,1]
        y_pred_threshold = []
        for i in y_pro:
            if i >= threshold:
                y_pred_threshold.append(1)
            else:
                y_pred_threshold.append(0)
        auc = metrics.roc_auc_score(y_test, y_pro)
        recall = metrics.recall_score(y_test, y_pred_threshold, average='binary')
        precision = metrics.recall_score(y_test, y_pred_threshold, average='binary')
        f1 = metrics.f1_score(y_test, y_pred_threshold, average='binary')
        con_mat = confusion_matrix(y_test, y_pred_threshold, labels=[0,1])
        acc = (con_mat[0,0]+con_mat[1,1])/sum(sum(con_mat))
        sen = (con_mat[1,1]/sum(con_mat[1,:]))
        spe = (con_mat[0,0]/sum(con_mat[0,:]))
        mcc = (con_mat[0,0]*con_mat[1,1]-con_mat[0,1]*con_mat[1,0])/((sum(con_mat[1,:])*sum(con_mat[0,:])*sum(con_mat[:,0])*sum(con_mat[:,1]))**0.5)
        listauc.append(auc)
        listrecall.append(recall)
        listprecision.append(precision)
        listf1.append(f1)
        listacc.append(acc)
        listsen.append(sen)
        listspe.append(spe)
        listmcc.append(mcc)
    aucmean = sum(listauc)/len(listauc)
    recallmean = sum(listrecall)/len(listrecall)
    precisionmean = sum(listprecision)/len(listprecision)
    f1mean = sum(listf1)/len(listf1)
    accmean = sum(listacc)/len(listacc)
    senmean = sum(listsen)/len(listsen)
    spemean = sum(listspe)/len(listspe)
    mccmean = sum(listmcc)/len(listmcc)

    df = pd.DataFrame({'aucmean':('%.4f' % aucmean), 'recallmean':('%.4f' % recallmean), 'precisionmean':('%.4f' % precisionmean),
                       'f1mean':('%.4f' % f1mean) , 'accmean':('%.4f' % accmean), 'senmean':('%.4f' % senmean),
                       'spemean':('%.4f' % spemean), 'mccmean':('%.4f' % mccmean)},index=[0])

    return df
    
    
def class2_threshold_test(model,X_train,y_train,X_test,y_test,thresholds):

    threshold = thresholds
    clf = model.fit(X_train,y_train)
    y_pro = clf.predict_proba(X_test)[:,1]
    y_pred_threshold = []
    for i in y_pro:
        if i >= threshold:
            y_pred_threshold.append(1)
        else:
            y_pred_threshold.append(0)
    auc = metrics.roc_auc_score(y_test, y_pro)
    recall = metrics.recall_score(y_test, y_pred_threshold, average='binary')
    precision = metrics.recall_score(y_test, y_pred_threshold, average='binary')
    f1 = metrics.f1_score(y_test, y_pred_threshold, average='binary')
    con_mat = confusion_matrix(y_test, y_pred_threshold, labels=[0,1])
    acc = (con_mat[0,0]+con_mat[1,1])/sum(sum(con_mat))
    sen = (con_mat[1,1]/sum(con_mat[1,:]))
    spe = (con_mat[0,0]/sum(con_mat[0,:]))
    mcc = (con_mat[0,0]*con_mat[1,1]-con_mat[0,1]*con_mat[1,0])/((sum(con_mat[1,:])*sum(con_mat[0,:])*sum(con_mat[:,0])*sum(con_mat[:,1]))**0.5)

    df = pd.DataFrame({'aucmean':('%.4f' % auc), 'recallmean':('%.4f' % recall), 'precisionmean':('%.4f' % precision),
                       'f1mean':('%.4f' % f1) , 'accmean':('%.4f' % acc), 'senmean':('%.4f' % sen),
                       'spemean':('%.4f' % spe), 'mccmean':('%.4f' % mcc)},index=[0])

    return df


def class2_test(model,X_train,y_train,X_test,y_test):

    clf = model.fit(X_train,y_train)
    y_pro = clf.predict_proba(X_test)[:,1]
    y_pred = clf.predict(X_test)
    auc = metrics.roc_auc_score(y_test, y_pro)
    recall = metrics.recall_score(y_test, y_pred, average='binary')
    precision = metrics.recall_score(y_test, y_pred, average='binary')
    f1 = metrics.f1_score(y_test, y_pred, average='binary')
    con_mat = confusion_matrix(y_test, y_pred, labels=[0,1])
    acc = (con_mat[0,0]+con_mat[1,1])/sum(sum(con_mat))
    sen = (con_mat[1,1]/sum(con_mat[1,:]))
    spe = (con_mat[0,0]/sum(con_mat[0,:]))
    mcc = (con_mat[0,0]*con_mat[1,1]-con_mat[0,1]*con_mat[1,0])/((sum(con_mat[1,:])*sum(con_mat[0,:])*sum(con_mat[:,0])*sum(con_mat[:,1]))**0.5)

    df = pd.DataFrame({'aucmean':('%.4f' % auc), 'recallmean':('%.4f' % recall), 'precisionmean':('%.4f' % precision),
                       'f1mean':('%.4f' % f1) , 'accmean':('%.4f' % acc), 'senmean':('%.4f' % sen),
                       'spemean':('%.4f' % spe), 'mccmean':('%.4f' % mcc)},index=[0])

    return df

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

def plotauc_2classes(y_test,y_score,title):
    
    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.figure()
    lw = 2
    roc_auc = auc(fpr,tpr)
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    
    return 
    
from itertools import cycle    

def plotauc_muclasses(y_test,y_score,title,nclass):
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nclass):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())#.ravel()平铺
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nclass)]))#拼接再去重复

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nclass):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nclass

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    #roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    roc_auc["macro"] = roc_auc_score(y_test,y_score,average='macro')

    # Plot all ROC curves
    plt.figure()
    lw = 2
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.4f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.4f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(nclass), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.4f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    
    return
    

from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.multiclass import OneVsOneClassifier


def list_round(lista,n):
    #对列表里的数字保留几位小数
    listb = []
    for i in range(len(lista)):
        listb.append(round(lista[i],n))
    
    return listb
    

def classmu_ovr_test_y(model, X_train, y_train, X_test, y_test, nclass, ovr):
    #多分类
    #OneVsRest
    #测试集结果
    #macro
    #y不采用onehot,      0,1,2,3,4...
    if nclass > 2:
        lb = preprocessing.LabelBinarizer()
        y_train_onehot = lb.fit_transform(y_train)
        y_test_onehot = lb.transform(y_test)
    elif nclass == 2:
        lb = preprocessing.LabelBinarizer()
        y_train_onehot0 = lb.fit_transform(y_train)
        y_test_onehot0 = lb.transform(y_test)
        y_train_onehot = np.hstack(((1-y_train_onehot0), y_train_onehot0))
        y_test_onehot = np.hstack(((1-y_test_onehot0), y_test_onehot0))
    
    ovr = ovr
    if ovr == 1:
        classifier = model
        classifier = OneVsRestClassifier(classifier).fit(X_train, y_train_onehot)
    elif ovr == 0:
        classifier = model
        classifier.fit(X_train, y_train)
    elif ovr == 2:
        classifier = model
        classifier = OneVsOneClassifier(classifier).fit(X_train, y_train)

    if hasattr(model, "decision_function") and nclass > 2: # use decision function
        prob_pos = classifier.decision_function(X_test)
        y_score2 = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())        
    else:  
        y_score2 = classifier.predict_proba(X_test)
    y_pred_onhot = (y_score2 == y_score2.max(axis=1)[:,None]).astype(int)
    y_pred = lb.inverse_transform(y_pred_onhot)
    matrix_ = confusion_matrix(y_test,y_pred)
    print("confusion_matrix=")
    print(matrix_)

    if ovr == 1:
        auc_proba_macro = metrics.roc_auc_score(y_test_onehot, y_score2, multi_class="ovr", average="macro")
        auc_proba_micro = metrics.roc_auc_score(y_test_onehot, y_score2, multi_class="ovr", average="micro")
    elif ovr == 0 and nclass ==2:
        auc_proba_macro = metrics.roc_auc_score(y_test, y_score2[:,1])
        auc_proba_micro = metrics.roc_auc_score(y_test, y_score2[:,1])
    elif ovr == 0 and nclass > 2:
        auc_proba_macro = metrics.roc_auc_score(y_test, y_score2, multi_class="ovr", average="macro")
        auc_proba_micro = metrics.roc_auc_score(y_test, y_score2, multi_class="ovr", average="weighted")
    elif ovr == 2:
        auc_proba_macro = metrics.roc_auc_score(y_test_onehot, y_score2, multi_class="ovo", average="macro")
        auc_proba_micro = metrics.roc_auc_score(y_test_onehot, y_score2, multi_class="ovo", average="micro")
    
    acc_all = ACCall(y_test,y_pred)
    print("acc_all=", acc_all)

    sens = sen(y_test,y_pred,nclass)
    spec = spe(y_test,y_pred,nclass)
    acc = ACC(y_test,y_pred,nclass)
    precision = pre(y_test,y_pred,nclass)

    recall = []
    f1_score=[]
    auc_one=[]
    precision2=[]
    for i in range(nclass):

        reca = metrics.recall_score(y_test_onehot[:,i], y_pred_onhot[:,i])
        recall.append(reca)

        f1 = metrics.f1_score(y_test_onehot[:,i], y_pred_onhot[:,i])
        f1_score.append(f1)

        auc_1=metrics.roc_auc_score(y_test_onehot[:,i], y_score2[:,i])
        auc_one.append(auc_1)

        preci=metrics.precision_score(y_test_onehot[:,i], y_pred_onhot[:,i])
        precision2.append(preci)

    index_name = []
    for i in range(nclass):
        index_name.append(i)
    df = pd.DataFrame({#'auc_decision_macro':(round(auc_decision_macro,4)), 'auc_decision_micro':(round(auc_decision_micro,4)),
                    'auc_proba_macro':(round(auc_proba_macro,4)),'auc_proba_micro':(round(auc_proba_micro,4)),'acc_all':(round(acc_all,4))
                   ,'sens':(list_round(sens,4)),'spec':(list_round(spec,4)),'acc':(list_round(acc,4)),'precision':(list_round(precision,4))
                   ,'recall':(list_round(recall,4)),'f1_score':(list_round(f1_score,4)),'auc_one':(list_round(auc_one,4)),'precision2':(list_round(precision2,4))}
                   ,index=index_name)
    return df, y_score2


def classmu_ovr_kfolds_y(model,Xtrain__,Ytrain__,nclasses,k,ovr):
    #多分类
    #ovr
    #训练集k折交叉验证
    #nclass类
    #y为0,1,2,3,4...
    ovr = ovr
    kf = KFold(n_splits=k, shuffle=False)
    listdf = []
    nclasses_ = nclasses
    for i, (train_index, test_index) in enumerate(kf.split(Xtrain__)):
        #print(f'KFold {i+1}:')
        #print("Train index:", train_index, "Test index:", test_index)
        X_train_, X_test_ = Xtrain__[train_index], Xtrain__[test_index]
        y_train_, y_test_ = Ytrain__[train_index], Ytrain__[test_index]
        df = classmu_ovr_test_y(model=model,X_train=X_train_,y_train=y_train_,X_test=X_test_,y_test=y_test_,nclass=nclasses_,ovr=ovr)[0]
        listdf.append(df)    
    df2 = sum(listdf)/k
  
    return df2


def classmu_ovr_GridSearch_y(model, X_train, y_train, X_test, y_test, nclass, ovr, parameter):#可以进行二分类或者多分类
    # 多分类或者二分类
    # OneVsRest
    # 测试集结果
    # macro
    # y不采用onehot,      0,1,2,3,4...
    if nclass > 2:
        lb = preprocessing.LabelBinarizer()
        y_train_onehot = lb.fit_transform(y_train)
        y_test_onehot = lb.transform(y_test)
    elif nclass == 2:
        lb = preprocessing.LabelBinarizer()
        y_train_onehot0 = lb.fit_transform(y_train)
        y_test_onehot0 = lb.transform(y_test)
        y_train_onehot = np.hstack(((1-y_train_onehot0), y_train_onehot0))
        y_test_onehot = np.hstack(((1-y_test_onehot0), y_test_onehot0))

    ovr = ovr
    if ovr == 1:
        classifiermodel = model
        classifier = OneVsRestClassifier(classifiermodel)
        GS = GridSearchCV(classifier, param_grid=parameter, cv=10)
        GS.fit(X_train, y_train_onehot)
        classifiermodel2 = model
        classifier_ = OneVsRestClassifier(classifiermodel2)
        classifier_.set_params(**GS.best_params_)
        classifier_.fit(X_train, y_train_onehot)

    elif ovr == 0:
        classifier = model
        if nclass == 2:
            GS = GridSearchCV(classifier, param_grid=parameter, cv=10, scoring='roc_auc')
        else:
            GS = GridSearchCV(classifier, param_grid=parameter, cv=10)
        GS.fit(X_train, y_train)
        classifier_ = model
        classifier_.set_params(**GS.best_params_)
        classifier_.fit(X_train, y_train)

    elif ovr == 2:
        classifiermodel = model
        classifier = OneVsOneClassifier(classifiermodel)
        GS = GridSearchCV(classifier, param_grid=parameter, cv=10)
        GS.fit(X_train, y_train)
        classifiermodel2 = model.copy
        classifier_ = OneVsOneClassifier(classifiermodel2)
        classifier_.set_params(**GS.best_params_)
        classifier_.fit(X_train, y_train)

    if hasattr(model, "decision_function") and nclass > 2:  # use decision function
        prob_pos = classifier_.decision_function(X_test)
        y_score2 = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        y_score2 = y_score2 / y_score2.sum(axis=1, keepdims=True)
    else:
        y_score2 = classifier_.predict_proba(X_test)
    y_pred_onhot = (y_score2 == y_score2.max(axis=1)[:, None]).astype(int)
    y_pred = lb.inverse_transform(y_pred_onhot)
    matrix_ = confusion_matrix(y_test, y_pred)
    print("confusion_matrix=")
    print(matrix_)
    print("classifier")
    print(classifier)
    print("GS.best_params_")
    print(GS.best_params_)
    print("GS.best_score_")
    print(GS.best_score_)

    if ovr == 1:
        auc_proba_macro = metrics.roc_auc_score(y_test_onehot, y_score2, multi_class="ovr", average="macro")
        auc_proba_micro = metrics.roc_auc_score(y_test_onehot, y_score2, multi_class="ovr", average="micro")
    elif ovr == 0 and nclass == 2:
        auc_proba_macro = metrics.roc_auc_score(y_test, y_score2[:,1])
        auc_proba_micro = metrics.roc_auc_score(y_test, y_score2[:,1])
    elif ovr == 0 and nclass > 2:
        auc_proba_macro = metrics.roc_auc_score(y_test, y_score2, multi_class="ovr", average="macro")
        auc_proba_micro = metrics.roc_auc_score(y_test, y_score2, multi_class="ovr", average="weighted")
    elif ovr == 2:
        auc_proba_macro = metrics.roc_auc_score(y_test_onehot, y_score2, multi_class="ovo", average="macro")
        auc_proba_micro = metrics.roc_auc_score(y_test_onehot, y_score2, multi_class="ovo", average="micro")

    acc_all = ACCall(y_test, y_pred)
    print("acc_all=", acc_all)

    sens = sen(y_test, y_pred, nclass)
    spec = spe(y_test, y_pred, nclass)
    acc = ACC(y_test, y_pred, nclass)
    precision = pre(y_test, y_pred, nclass)
    recall = []
    f1_score = []
    auc_one = []
    precision2 = []
    for i in range(nclass):
        reca = metrics.recall_score(y_test_onehot[:, i], y_pred_onhot[:, i])
        recall.append(reca)

        f1 = metrics.f1_score(y_test_onehot[:, i], y_pred_onhot[:, i])
        f1_score.append(f1)

        auc_1 = metrics.roc_auc_score(y_test_onehot[:, i], y_score2[:, i])
        auc_one.append(auc_1)

        preci = metrics.precision_score(y_test_onehot[:, i], y_pred_onhot[:, i])
        precision2.append(preci)

    index_name = []
    for i in range(nclass):
        index_name.append(i)
    df = pd.DataFrame(
        {  # 'auc_decision_macro':(round(auc_decision_macro,4)), 'auc_decision_micro':(round(auc_decision_micro,4)),
            'auc_proba_macro': (round(auc_proba_macro, 4)), 'auc_proba_micro': (round(auc_proba_micro, 4)),
            'acc_all': (round(acc_all, 4))
            , 'sens': (list_round(sens, 4)), 'spec': (list_round(spec, 4)), 'acc': (list_round(acc, 4)),
            'precision': (list_round(precision, 4))
            , 'recall': (list_round(recall, 4)), 'f1_score': (list_round(f1_score, 4)),
            'auc_one': (list_round(auc_one, 4)), 'precision2': (list_round(precision2, 4))}
        , index=index_name)
    return df, GS.best_params_

def onehottocontin(y_onehot,nclass):
    #array([[0, 0, 1],[0, 1, 0]]) to array([2, 1])
    #独热编码转化成非独热编码
    #y_onehot为独热编码
    y_continu = np.zeros(shape=(y_onehot.shape[0],))
    for i in range(len(y_onehot)):
        for j in range(nclass):
            if y_onehot[i][j] ==1:
                
                y_continu[i] = j
    return y_continu
    

def classmu_ovr_test_yonehot(model,X_train,y_train_onehot,X_test,y_test_onehot,nclass,ovr):
    #多分类
    #OneVsRest
    #测试集结果
    #macro
    #y采用onehot
    y_train = onehottocontin(y_train_onehot,nclass)
    y_test = onehottocontin(y_test_onehot,nclass)
    ovr = ovr
    if ovr == 1:
        classifier = OneVsRestClassifier(model).fit(X_train, y_train_onehot)
    elif ovr == 0:
        classifier = model.fit(X_train, y_train)
    if ovr == 2:
        classifier = OneVsOneClassifier(model).fit(X_train, y_train)

    if hasattr(model, "decision_function"):# use decision function
        prob_pos = classifier.predict_proba(X_test)
        y_score2 = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())        
    else:  
        y_score2 = classifier.predict_proba(X_test)

    y_pred_onhot = (y_score2 == y_score2.max(axis=1)[:, None]).astype(int)
    y_pred = onehottocontin(y_pred_onhot, nclass)

    matrix_ = confusion_matrix(y_test, y_pred)
    print("confusion_matrix=")
    print(matrix_)

    if ovr == 1:
        auc_proba_macro = metrics.roc_auc_score(y_test_onehot, y_score2, multi_class="ovr", average="macro")
        auc_proba_micro = metrics.roc_auc_score(y_test_onehot, y_score2, multi_class="ovr", average="micro")
    elif ovr == 0:
        auc_proba_macro = metrics.roc_auc_score(y_test, y_score2, multi_class="ovr", average="macro")
        auc_proba_micro = metrics.roc_auc_score(y_test, y_score2, multi_class="ovr", average="micro")
    elif ovr == 2:
        auc_proba_macro = metrics.roc_auc_score(y_test_onehot, y_score2, multi_class="ovo", average="macro")
        auc_proba_micro = metrics.roc_auc_score(y_test_onehot, y_score2, multi_class="ovo", average="micro")

    acc_all = ACCall(y_test, y_pred)
    print("acc_all=", acc_all)

    sens = sen(y_test, y_pred, nclass)
    spec = spe(y_test, y_pred, nclass)
    acc = ACC(y_test, y_pred, nclass)
    precision = pre(y_test, y_pred, nclass)
    recall = []
    f1_score = []
    auc_one = []
    precision2 = []
    for i in range(nclass):
        reca = metrics.recall_score(y_test_onehot[:, i], y_pred_onhot[:, i])
        recall.append(reca)

        f1 = metrics.f1_score(y_test_onehot[:, i], y_pred_onhot[:, i])
        f1_score.append(f1)

        auc_1 = metrics.roc_auc_score(y_test_onehot[:, i], y_score2[:, i])
        auc_one.append(auc_1)

        preci = metrics.precision_score(y_test_onehot[:, i], y_pred_onhot[:, i])
        precision2.append(preci)

    index_name = []
    for i in range(nclass):
        index_name.append(i)
    df = pd.DataFrame(
        {  # 'auc_decision_macro':(round(auc_decision_macro,4)), 'auc_decision_micro':(round(auc_decision_micro,4)),
            'auc_proba_macro': (round(auc_proba_macro, 4)), 'auc_proba_micro': (round(auc_proba_micro, 4)),
            'acc_all': (round(acc_all, 4))
            , 'sens': (list_round(sens, 4)), 'spec': (list_round(spec, 4)), 'acc': (list_round(acc, 4)),
            'precision': (list_round(precision, 4))
            , 'recall': (list_round(recall, 4)), 'f1_score': (list_round(f1_score, 4)),
            'auc_one': (list_round(auc_one, 4)), 'precision2': (list_round(precision2, 4))}
        , index=index_name)

    return df 


def classmu_ovr_kfolds_yonehot(model,Xtrain__,Ytrain_onehot__,nclasses,k,ovr):
    #多分类
    #ovr
    #训练集k折交叉验证
    #nclass类
    #y为0,1,2,3,4...
    ovr=ovr
    kf = KFold(n_splits=k, shuffle=False)
    listdf = []
    nclasses_ = nclasses
    for i, (train_index, test_index) in enumerate(kf.split(Xtrain__)):
        #print(f'KFold {i+1}:')
        #print("Train index:", train_index, "Test index:", test_index)
        X_train_, X_test_ = Xtrain__[train_index], Xtrain__[test_index]
        y_train_onehot_, y_test_onehot_ = Ytrain_onehot__[train_index], Ytrain_onehot__[test_index]
        
        df = classmu_ovr_test_yonehot(model=model,X_train=X_train_,y_train_onehot=y_train_onehot_,X_test=X_test_,y_test_onehot=y_test_onehot_,nclass=nclasses_,ovr=ovr)
        listdf.append(df)    
    df2 = sum(listdf)/k
  
    return df2

def find_best_threshold_sum(y_true, y_prob):
    """根据输入的真实标签和预测概率值，调整阈值并返回最优阈值、最大敏感性和特异性之和的值"""
    # 初始化最优阈值、最大敏感性和特异性之和的值
    best_threshold = -1
    best_sensitivity_specificity_sum = -1
    # 根据预测概率值和真实标签，计算不同阈值下的混淆矩阵
    for threshold in np.arange(0.3, 0.7, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        sensitivity_specificity_sum = sensitivity + specificity
        # 更新最优阈值和最大敏感性和特异性之和的值
        if sensitivity_specificity_sum > best_sensitivity_specificity_sum:
            best_sensitivity_specificity_sum = sensitivity_specificity_sum
            best_threshold = threshold
    return best_threshold, best_sensitivity_specificity_sum

def ensemble_model_vote(y_test_onehot,listmodels,vote,ovr):
    """
    ytest_onehot:m个样本n个分类onehot
    list包含k个分类器的预测n个分类的预测概率
    返回集成模型的混淆矩阵
    Hard少数服从多数,Soft概率相加
    """
    nclass = len(listmodels[0][0])
    models_pred = np.zeros(shape=(listmodels[0].shape[0], listmodels[0].shape[1]), dtype=float, order='C')

    if vote == "Hard":
        for i in range(len(listmodels)):
            model_prob_i = np.nan_to_num(listmodels[i])
            model_pred_i = (model_prob_i == model_prob_i.max(axis=1)[:, None]).astype(int)
            models_pred = models_pred+model_pred_i
        models_pred = models_pred/nclass

    elif vote == "Soft":
        for i in range(len(listmodels)):
            model_prob_i = np.nan_to_num(listmodels[i])
            for j in range(model_prob_i.shape[0]):
                if sum(model_prob_i[j]) ==0:
                    model_prob_i[j] = model_prob_i[j]
                else:
                    model_prob_i[j] = model_prob_i[j]/sum(model_prob_i[j])
            models_pred = models_pred + model_prob_i
        models_pred = models_pred/nclass
    y_score2 = models_pred
    y_test = onehottocontin(y_test_onehot, nclass=nclass)
    y_pred_onhot = (y_score2 == y_score2.max(axis=1)[:, None]).astype(int)
    y_pred = onehottocontin(y_pred_onhot, nclass)

    matrix_ = confusion_matrix(y_test, y_pred)
    print("confusion_matrix=")
    print(matrix_)

    if ovr == 1:
        auc_proba_macro = metrics.roc_auc_score(y_test_onehot, y_score2, multi_class="ovr", average="macro")
        auc_proba_micro = metrics.roc_auc_score(y_test_onehot, y_score2, multi_class="ovr", average="micro")
    elif ovr == 0 and nclass == 2:
        auc_proba_macro = metrics.roc_auc_score(y_test, y_score2[:,1])
        auc_proba_micro = metrics.roc_auc_score(y_test, y_score2[:,1])
    elif ovr == 0 and nclass > 2:
        # auc_proba_macro = metrics.roc_auc_score(y_test, y_score2, multi_class="ovr", average="macro")
        # auc_proba_micro = metrics.roc_auc_score(y_test, y_score2, multi_class="ovr", average="micro")
        auc_proba_macro = metrics.roc_auc_score(y_test_onehot, y_score2, multi_class="ovr", average="macro")
        auc_proba_micro = metrics.roc_auc_score(y_test_onehot, y_score2, multi_class="ovr", average="micro")
    elif ovr == 2:
        auc_proba_macro = metrics.roc_auc_score(y_test_onehot, y_score2, multi_class="ovo", average="macro")
        auc_proba_micro = metrics.roc_auc_score(y_test_onehot, y_score2, multi_class="ovo", average="micro")

    acc_all = ACCall(y_test, y_pred)
    print("acc_all=", acc_all)

    sens = sen(y_test, y_pred, nclass)
    spec = spe(y_test, y_pred, nclass)
    acc = ACC(y_test, y_pred, nclass)
    precision = pre(y_test, y_pred, nclass)
    recall = []
    f1_score = []
    auc_one = []
    precision2 = []
    for i in range(nclass):
        reca = metrics.recall_score(y_test_onehot[:, i], y_pred_onhot[:, i])
        recall.append(reca)

        f1 = metrics.f1_score(y_test_onehot[:, i], y_pred_onhot[:, i])
        f1_score.append(f1)

        auc_1 = metrics.roc_auc_score(y_test_onehot[:, i], y_score2[:, i])
        auc_one.append(auc_1)

        preci = metrics.precision_score(y_test_onehot[:, i], y_pred_onhot[:, i])
        precision2.append(preci)

    index_name = []
    for i in range(nclass):
        index_name.append(i)
    df = pd.DataFrame(
        {  # 'auc_decision_macro':(round(auc_decision_macro,4)), 'auc_decision_micro':(round(auc_decision_micro,4)),
            'auc_proba_macro': (round(auc_proba_macro, 4)), 'auc_proba_micro': (round(auc_proba_micro, 4)),
            'acc_all': (round(acc_all, 4))
            , 'sens': (list_round(sens, 4)), 'spec': (list_round(spec, 4)), 'acc': (list_round(acc, 4)),
            'precision': (list_round(precision, 4))
            , 'recall': (list_round(recall, 4)), 'f1_score': (list_round(f1_score, 4)),
            'auc_one': (list_round(auc_one, 4)), 'precision2': (list_round(precision2, 4))}
        , index=index_name)

    return df


def ensemble_model_blending(model, y_valid_onehot, y_test_onehot, prob_valid_list, prob_test_list, ovr, parameter):
    """
    ytest_onehot:m个样本n个分类onehot
    list包含k个分类器的预测n个分类的预测概率
    返回集成模型的混淆矩阵
    嵌套一层模型
    """
    model_blending = model
    nclass = y_test_onehot.shape[1]
    y_valid = onehottocontin(y_valid_onehot, nclass=nclass)
    y_test = onehottocontin(y_test_onehot, nclass)

    prob_valid_merge = np.nan_to_num(prob_valid_list[0])
    for i in range(1, len(prob_valid_list)):
        prob_valid_merge = np.hstack((prob_valid_merge, np.nan_to_num(prob_valid_list[i])))

    prob_test_merge = np.nan_to_num(prob_test_list[0])
    for i in range(1, len(prob_test_list)):
        prob_test_merge = np.hstack((prob_test_merge, np.nan_to_num(prob_test_list[i])))

    df, GS_best_params_ = classmu_ovr_GridSearch_y(model, X_train=prob_valid_merge, y_train=y_valid, X_test=prob_test_merge, y_test=y_test, nclass=nclass, ovr=ovr, parameter=parameter)

    if ovr == 0:
        model_blending = model_blending.set_params(**GS_best_params_)
        model_blending.fit(prob_valid_merge, y_valid)
    elif ovr != 0:
        model_blending = OneVsOneClassifier(model_blending)
        model_blending.set_params(**GS_best_params_)
        model_blending.fit(prob_valid_merge, y_valid)
    return df, GS_best_params_, model_blending

def ensemble_model_blending2(model, y_valid_onehot, y_test_onehot, prob_valid_list, prob_test_list, ovr):
    """
    ytest_onehot:m个样本n个分类onehot
    list包含k个分类器的预测n个分类的预测概率
    返回集成模型的混淆矩阵
    嵌套一层模型
    """
    model_blending = model
    nclass = y_test_onehot.shape[1]
    y_valid = onehottocontin(y_valid_onehot, nclass=nclass)
    y_test = onehottocontin(y_test_onehot, nclass)

    prob_valid_merge = np.nan_to_num(prob_valid_list[0])
    for i in range(1, len(prob_valid_list)):
        prob_valid_merge = np.hstack((prob_valid_merge, np.nan_to_num(prob_valid_list[i])))

    prob_test_merge = np.nan_to_num(prob_test_list[0])
    for i in range(1, len(prob_test_list)):
        prob_test_merge = np.hstack((prob_test_merge, np.nan_to_num(prob_test_list[i])))

    df, y_prob_test = classmu_ovr_test_y(model, X_train=prob_valid_merge, y_train=y_valid, X_test=prob_test_merge, y_test=y_test, nclass=nclass, ovr=ovr)

    if ovr == 0:
        model_blending.fit(prob_valid_merge, y_valid)
    elif ovr != 0:
        model_blending = OneVsOneClassifier(model_blending)
        model_blending.fit(prob_valid_merge, y_valid)
    return df, y_prob_test, model_blending


def classmu_yprobytest_toauc(yprob,y_test_onehot,nclass):
    #非onehot多分类可用roc_auc_score(y_test, y_score2, multi_class="ovr", average="macro")
    y_test = onehottocontin(y_test_onehot,nclass=y_test_onehot.shape[1])
    y_score2 = yprob
    y_pred_onhot = (y_score2 == y_score2.max(axis=1)[:, None]).astype(int)
    y_pred = onehottocontin(y_pred_onhot, nclass)

    matrix_ = confusion_matrix(y_test, y_pred)
    print("confusion_matrix=")
    print(matrix_)


    auc_proba_macro = roc_auc_score(y_test_onehot, y_score2, multi_class="ovr", average="macro")
    auc_proba_micro = roc_auc_score(y_test_onehot, y_score2, multi_class="ovr", average="micro")

    acc_all = ACCall(y_test, y_pred)
    print("acc_all=", acc_all)

    sens = sen(y_test, y_pred, nclass)
    spec = spe(y_test, y_pred, nclass)
    acc = ACC(y_test, y_pred, nclass)
    precision = pre(y_test, y_pred, nclass)
    recall = []
    f1_score = []
    auc_one = []
    precision2 = []
    for i in range(nclass):
        reca = metrics.recall_score(y_test_onehot[:, i], y_pred_onhot[:, i])
        recall.append(reca)

        f1 = metrics.f1_score(y_test_onehot[:, i], y_pred_onhot[:, i])
        f1_score.append(f1)

        auc_1 = metrics.roc_auc_score(y_test_onehot[:, i], y_score2[:, i])
        auc_one.append(auc_1)

        preci = metrics.precision_score(y_test_onehot[:, i], y_pred_onhot[:, i])
        precision2.append(preci)

    index_name = []
    for i in range(nclass):
        index_name.append(i)
    df = pd.DataFrame(
        {  # 'auc_decision_macro':(round(auc_decision_macro,4)), 'auc_decision_micro':(round(auc_decision_micro,4)),
            'auc_proba_macro': (round(auc_proba_macro, 4)), 'auc_proba_micro': (round(auc_proba_micro, 4)),
            'acc_all': (round(acc_all, 4))
            , 'sens': (list_round(sens, 4)), 'spec': (list_round(spec, 4)), 'acc': (list_round(acc, 4)),
            'precision': (list_round(precision, 4))
            , 'recall': (list_round(recall, 4)), 'f1_score': (list_round(f1_score, 4)),
            'auc_one': (list_round(auc_one, 4)), 'precision2': (list_round(precision2, 4))}
        , index=index_name)

    return df     


from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator,title
                        , X, y,#特征矩阵和标签
                       ax, #选择子图
                       ylim=None, #设置纵坐标的取值范围
                       cv=None, #交叉验证
                       n_jobs=None #设定所要使用的线程
                      ):
    """
    estimator:分类器
    learning_curve：绘制关于训练样本增加的学习曲线
    """
    cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y
                                                          ,cv=cv,n_jobs=n_jobs)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid() #显示网格作为背景，不是必须
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-'
          , color="r",label="Training score")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-'
          , color="g",label="Test score")
    ax.legend(loc="best")
    return ax

# ########################################################################plot_learning_curve使用范例：
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier as RFC
# from sklearn.tree import DecisionTreeClassifier as DTC
# from sklearn.linear_model import LogisticRegression as LR
# from time import time
# import datetime
# title = ["Naive Bayes","DecisionTree","SVM, RBF kernel","RandomForest","Logistic"]
# model = [GaussianNB(),DTC(),SVC(gamma=0.001)
#          ,RFC(n_estimators=50),LR(C=.1,solver="lbfgs")]
# #cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
# fig, axes = plt.subplots(1,5,figsize=(30,6))
# for ind, title_, estimator in zip(range(len(title)),title,model):
#     times = time()
#     plot_learning_curve(estimator, title_, X, y,
#                         ax=axes[ind], ylim = [0.7, 1.05],n_jobs=4, cv=cv)
#     print("{}:{}".format(title_,datetime.datetime.fromtimestamp(time()-times).strftime("%M:%S:%f")))
# plt.show()
# #[*zip(range(len(title)),title,model)]#查看zip封装的隐形





















    
    
   


