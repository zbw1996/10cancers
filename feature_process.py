def RobustRank(df_import):
    """
    RobustRank: 求多个排序的综合排序
    df_import：多个模型的特征排序构成的数据框，可以是字符串形式
    df_result:
    """
    import pandas as pd
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_from_pd_df = robjects.conversion.py2rpy(df_import)
    robjects.r('''
    RRA <- function(r_from_pd_df) {
      library(RobustRankAggreg);
      glist <- list();
      for (i in colnames(r_from_pd_df)) {glist[[i]] <- as.character(r_from_pd_df[,i])};
      freq=as.data.frame(table(unlist(glist)));
      ag=aggregateRanks(glist);
      ag$Freq=freq[match(ag$Name,freq$Var1),2];
      return (ag)
    }
        ''')
    ag = robjects.r['RRA'](r_from_pd_df)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df_result = robjects.conversion.rpy2py(ag)

    return list(df_result['Name'])











