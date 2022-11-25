# Importing of library and data
library(scorecard)
data("germancredit")

# Filtering of variables with >= iv_limit = 0.02, <= missing_limit = 0.95 
# and <= identical_limit_limit = 0.95
data_f.df = var_filter(germancredit, y="creditability")

# Splitting data into train and test data with ratio 0.75
data_f.list = split_df(data_f.df,"creditability",ratio=c(0.75,0.25))
class(data_f.list)
lapply(data_f.list,class)
lapply(data_f.list, dim)

# Generation a list for response variable: Dummy variable for defaults
default.list = lapply(data_f.list, function(x) x$creditability)

# Binning of train data
# breaks.list is saved and imported seperately
bins.list = woebin(data_f.list$train,
                   "creditability",
                   save_breaks_list = "breaks.list")
head(bins.list)
woebin_plot (bins.list)

# WOE-Transformation of train and test data
data_woe.list = lapply(data_f.list,
                       function(x) woebin_ply(x, bins.list))
lapply(data_woe.list, class)
lapply(data_woe.list, dim)

head(data_woe.list$train)

# Bin-Group (GRP) Transformation of train and test data
data_grp.list = lapply(data_f.list,
                       function(x) woebin_ply(x, bins.list, to = 'bin'))
lapply(data_grp.list, class)
lapply(data_grp.list, dim)

# 1. Creating Credit Risk Scorecard with the left 14 WOE-transformed predictor variables
data_woe_first_iteration.glm <- glm(creditability ~ .,
                                    family = binomial(),
                                    data = data_woe.list$train)
summary(data_woe_first_iteration.glm)
scorecard_woe_first_iteration.scm <- scorecard(bins.list,
                                               data_woe_first_iteration.glm)

score_woe_first_iteration.df = scorecard_ply(germancredit,
                                             scorecard_woe_first_iteration.scm,
                                             only_total_score = FALSE)

score_woe_first_iteration.list <- lapply(data_f.list,
                                         function(x) scorecard_ply(x, scorecard_woe_first_iteration.scm))

y<-"creditability"
x<-c("status.of.existing.checking.account","duration.in.month","credit.history",
     "purpose","credit.amount","savings.account.and.bonds","present.employment.since",
     "installment.rate.in.percentage.of.disposable.income","other.debtors.or.guarantors",
     "property","age.in.years","other.installment.plans","housing")
breaks.list = source("breaks.list.R")

report(data_f.list,
       y,
       x,
       breaks.list,
       seed = NULL,
       save_report = "Report_WOE_first_iteration")
# Assessing the Out-of-Sample Gini coefficient
# -> Gini train = 
# -> Gini test = 

# Choosing of 7 Variables with lowest Significance Value

# Validation of chosen variables with GRP transformed predictor values
data_grp_first_iteration.glm <- glm(creditability ~ .,
                                    family = binomial(),
                                    data = data_grp.list$train)
summary(data_grp_first_iteration.glm)

# 2. Creating a Credit Risk Scorecard with filtered 7 predictor variables
# WOE transformed predictor variables
data_woe_second_iteration.glm <- glm(creditability ~ status.of.existing.checking.account_woe
                                     +duration.in.month_woe
                                     +credit.history_woe+purpose_woe+credit.amount_woe
                                     +savings.account.and.bonds_woe+age.in.years_woe,
                                    family = binomial(),
                                    data = data_woe.list$train)
summary(data_woe_second_iteration.glm)
scorecard_woe_second_iteration.scm <- scorecard(bins.list,
                                               data_woe_second_iteration.glm)

score_woe_second_iteration.df = scorecard_ply(germancredit,
                                             scorecard_woe_second_iteration.scm,
                                             only_total_score = FALSE)

score_woe_second_iteration.list <- lapply(data_f.list,
                                         function(x) scorecard_ply(x, scorecard_woe_second_iteration.scm))
# Validation of WOE transformed Scoreboard
# Probability prediction of train and test samples
predProb_woe.list <- lapply(data_woe.list,
                        function(x) predict(data_woe_second_iteration.glm,
                                            type = 'response',
                                            x)
)

# Prediction of scorecards for the sub-samples
head(score_woe_second_iteration.list$train)
head(score_woe_second_iteration.list$test)

# Prediction accuracy (multiple dataset): IS- and OoS-Testing in one
# Predicted Probabilities
perf_eva(pred = predProb_woe.list,
         label = default.list,
         binomial_metric = c("gini","auc","r2", "rmse"),
         show_plot=c("roc","ks"),
         confusion_matrix = TRUE)
# Predicted Scores
perf_eva(pred = score_woe_second_iteration.list,
         label = default.list,
         binomial_metric = c("gini","auc","r2", "rmse"),
         show_plot=c("roc","ks"),
         confusion_matrix = TRUE)

# Report of Scoreboard with WOE-transformed predictor variables
y2<-"creditability"
x2<-c("status.of.existing.checking.account","duration.in.month","credit.history",
      "purpose","credit.amount","savings.account.and.bonds",
      "age.in.years")

report(data_f.list,
       y2,
       x2,
       breaks.list,
       seed = NULL,
       save_report = "Report_WOE_second_iteration")
# Assessing Out-of-Sample Gini coefficient
# -> Gini train = 
# -> Gini test = 

# 2. Creating a Credit Risk Scorecard with filtered 7 predictor variables
# Group Bin (GRP) transformed predictor variables
data_grp_second_iteration.glm <- glm(creditability ~ status.of.existing.checking.account_bin
                                     +duration.in.month_bin
                                     +credit.history_bin+purpose_bin+credit.amount_bin
                                     +savings.account.and.bonds_bin+age.in.years_bin,
                                     family = binomial(),
                                     data = data_grp.list$train)
summary(data_grp_second_iteration.glm)

# Creation of Scorecard is not possible for the GRP tranformed predictor variables

# Validation of GRP transformed logistic regression
# Probability prediction of train and test samples
predProb_grp.list <- lapply(data_grp.list,
                            function(x) predict(data_grp_second_iteration.glm,
                                                type = 'response',
                                                x)
)

# Prediction accuracy (multiple dataset): IS- and OoS-Testing in one
# Predicted Probabilities
perf_eva(pred = predProb_grp.list,
         label = default.list,
         binomial_metric = c("gini","auc","r2", "rmse"),
         show_plot=c("roc","ks"),
         confusion_matrix = TRUE)
