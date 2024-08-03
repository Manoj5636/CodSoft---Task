# Load necessary libraries
library(caret)
library(ggplot2)
library(dplyr)

# Load the Iris dataset
data(iris)
str(iris)

##  Data Preprocessing
# Check for missing values
sum(is.na(iris))

# Summary statistics
summary(iris)

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
trainData <- iris[trainIndex, ]
testData <- iris[-trainIndex, ]

# Feature Scaling (Standardization)
preProcessValues <- preProcess(trainData[, 1:4], method = c("center", "scale"))
trainDataScaled <- predict(preProcessValues, trainData[, 1:4])
testDataScaled <- predict(preProcessValues, testData[, 1:4])

# Combine scaled features with target variable
trainDataScaled <- cbind(trainDataScaled, Species = trainData$Species)
testDataScaled <- cbind(testDataScaled, Species = testData$Species)

## Model Training
# Train a Random Forest model
rfModel <- train(Species ~ ., data = trainDataScaled, method = "rf", 
                 trControl = trainControl(method = "cv", number = 10),
                 importance = TRUE)

## Model Evaluation
# Predictions using Random Forest
rfPredictions <- predict(rfModel, testDataScaled)
rfConfusionMatrix <- confusionMatrix(rfPredictions, testDataScaled$Species)
print("Random Forest Model Accuracy:")
print(rfConfusionMatrix)

# Feature Importance from Random Forest
importancePlot <- varImp(rfModel, scale = FALSE)
print(importancePlot)

# Plot Feature Importance
ggplot(importance_df, aes(x = reorder(Feature, Overall), y = Overall)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  xlab("Features") +
  ylab("Importance") +
  ggtitle("Feature Importance from Random Forest Model")

# Scatter Plot of Sepal Length vs Sepal Width
ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) +
  geom_point(size = 3) +
  labs(title = "Sepal Length vs Sepal Width by Species") +
  theme_minimal()

# Scatter Plot of Petal Length vs Petal Width
ggplot(iris, aes(x = Petal.Length, y = Petal.Width, color = Species)) +
  geom_point(size = 3) +
  labs(title = "Petal Length vs Petal Width by Species") +
  theme_minimal()

