import Foundation
import CreateML

// Specify Data
let trainingCSV = URL(fileURLWithPath: "/Users/noemiquezada/Documents/playgrounds/BlogArticleCategoryClassifier/PostsTrainingData.csv")
let testCSV = URL(fileURLWithPath: "/Users/noemiquezada/Documents/playgrounds/BlogArticleCategoryClassifier/PostsTestData.csv")
let trainingData = try MLDataTable(contentsOf: trainingCSV)

print(trainingData)

let testData = try MLDataTable(contentsOf: testCSV)

print(testData)

// Create Model
let model = try MLTextClassifier(trainingData: trainingData, textColumn: "title", labelColumn: "topic")

// Evaluate Model
let result = model.evaluation(on: testData)
print (result)
let writeToUrl = URL(fileURLWithPath: "/Users/noemiquezada/Documents/playgrounds/BlogArticleCategoryClassifier/BlogArticleCategoryClassifier.mlmodel")

try model.write(to: writeToUrl)

