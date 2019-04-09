namespace Demo.SentimentAnalysis.Part1

open Microsoft.ML

open Demo.SentimentAnalysis.Part1.Models
open Demo.SentimentAnalysis.Part1.Display

module Training = 
    let trainModel =
        let context = MLContext()

        let loader = context.Data.CreateTextLoader<SentimentData>(separatorChar = '\t', hasHeader = false)
               
        let dataFile = "./data/imdb_labelled.txt"
        let allData = loader.Load(dataFile)
        let splitData = context.Clustering.TrainTestSplit(allData, testFraction=0.3)

        let pipeline = 
            context
                .Transforms.Text.FeaturizeText("Features", "SentimentText")
                .Append(context.BinaryClassification.Trainers.FastForest())

        let model = pipeline.Fit(splitData.TrainSet)

        displayEvaluation context model "Train" splitData.TrainSet
        displayEvaluation context model "Test" splitData.TestSet

        (context, model)