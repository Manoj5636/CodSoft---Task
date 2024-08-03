# Loading Necessary Libraries
library(tidyverse)
library(caret)
library(randomForest)
library(rpart.plot)
library(ggplot2)

# Load the Titanic dataset:
titanic <- read.csv("C:/Users/Manoj Kumar S/OneDrive/Documents/CodSoft Tasks/Titanic-Dataset.csv")
titanic

# Exploratory Data Analysis
str(titanic)
summary(titanic)
head(titanic)

## Data Cleaning and Feature Engineering 
# Handling missing values
titanic$Age[is.na(titanic$Age)] <- median(titanic$Age, na.rm = TRUE)
titanic$Embarked[is.na(titanic$Embarked)] <- 'S'
titanic$Fare[is.na(titanic$Fare)] <- median(titanic$Fare, na.rm = TRUE)
titanic$Cabin[is.na(titanic$Cabin)] <- 'Unknown'

# Convert categorical variables to factors
titanic$Survived <- factor(titanic$Survived, levels = c(0, 1))
titanic$Pclass <- factor(titanic$Pclass, levels = c(1, 2, 3))
titanic$Sex <- factor(titanic$Sex, levels = c('male', 'female'))
titanic$Embarked <- factor(titanic$Embarked, levels = c('C', 'Q', 'S'))

# Feature Engineering
titanic$FamilySize <- titanic$SibSp + titanic$Parch + 1
titanic$IsAlone <- ifelse(titanic$FamilySize == 1, 1, 0)
titanic$Title <- sapply(titanic$Name, function(x) strsplit(x, split = '[,.]')[[1]][2])
titanic$Title <- sub(' ', '', titanic$Title)
titanic$Title <- factor(titanic$Title)

# Drop irrelevant columns
titanic <- titanic %>% select(-PassengerId, -Name, -Ticket, -Cabin)
titanic

# Data Splitting
set.seed(123)
trainIndex <- createDataPartition(titanic$Survived, p = 0.8, list = FALSE)
train <- titanic[trainIndex, ]
test <- titanic[-trainIndex, ]

## Model Building 
# Random Forest Model
rf_model <- randomForest(Survived ~ ., data = train, ntree = 500, mtry = 3, importance = TRUE)
rf_predictions <- predict(rf_model, test)
rf_predictions

## Model Evaluation
# Confusion Matrices
rf_cm <- confusionMatrix(rf_predictions, test$Survived)
print(rf_cm)

# Variable Importance Plot for Random Forest
var_importance <- data.frame(Variable = rownames(importance(rf_model)), Importance = importance(rf_model)[, 1])

ggplot(var_importance, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Variable Importance from Random Forest Model", x = "Variables", y = "Importance")

# Distribution of Survived vs Not Survived
ggplot(train, aes(x = Survived)) +
  geom_bar(aes(fill = Survived), position = "dodge") +
  theme_minimal() +
  labs(title = "Distribution of Survival", x = "Survived", y = "Count")

# Age Distribution by Survival
ggplot(train, aes(x = Age, fill = Survived)) +
  geom_histogram(binwidth = 5, position = "dodge") +
  theme_minimal() +
  labs(title = "Age Distribution by Survival", x = "Age", y = "Count")

# Survival Rate by Sex
ggplot(train, aes(x = Sex, fill = Survived)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  labs(title = "Survival Rate by Sex", x = "Sex", y = "Proportion")

# Survival Rate by Class
ggplot(train, aes(x = Pclass, fill = Survived)) +
  geom_bar(position = "fill") +
  theme_minimal() +
  labs(title = "Survival Rate by Class", x = "Pclass", y = "Proportion")



