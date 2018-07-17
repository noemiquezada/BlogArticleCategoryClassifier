import Foundation
import CreateML

// Import Training Data
let trainingCSV = URL(fileURLWithPath: "/Users/noemiquezada/Documents/playgrounds/BlogArticleCategoryClassifier/PostsTrainingData.csv")
let trainingData = try MLDataTable(contentsOf: trainingCSV)

// Create Model
let model = try MLTextClassifier(trainingData: trainingData, textColumn: "title", labelColumn: "topic")

// Import Test Data
let testCSV = URL(fileURLWithPath: "/Users/noemiquezada/Documents/playgrounds/BlogArticleCategoryClassifier/PostsTestData.csv")
let testData = try MLDataTable(contentsOf: testCSV)

// Evaluate Model
let result = model.evaluation(on: testData)
print (result)
let writeToUrl = URL(fileURLWithPath: "/Users/noemiquezada/Documents/playgrounds/BlogArticleCategoryClassifier/BlogArticleCategoryClassifier.mlmodel")

// Create ML Model
try model.write(to: writeToUrl)

