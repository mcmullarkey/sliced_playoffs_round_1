library(tidyverse)
library(tidymodels)
library(doMC)
library(finetune)
library(beepr)
library(treesnip)
library(catboost)
library(lightgbm)

mlb_folds <- read_rds("mlb_folds.rds")
sliced_set <- read_rds("sliced_set.rds")

doMC::registerDoMC(cores = 7)

## Time to tune with cross-validation

race <- sliced_set %>% 
  workflow_map(
    "tune_race_anova",
    grid = 6,
    seed = 33,
    resamples = mlb_folds,
    metrics = metric_set(mn_log_loss),
    control = control_race(
      save_pred = TRUE,
      save_workflow = FALSE,
      parallel_over = "everything"
    )
  )
beep()