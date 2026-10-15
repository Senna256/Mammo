library(tidyverse)


df = read.csv("/home/enric/Datasets/Original/vindr/finding_annotations.csv")

colnames(df)

a <- table(df$split, df$finding_categories)
View(a)

