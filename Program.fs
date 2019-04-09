open System.IO
open Microsoft.ML
open Microsoft.ML.Data

open Demo.SentimentAnalysis.Models
open Microsoft.ML.Data
open Microsoft.ML.Trainers.FastTree

let displayEvaluation (ml : MLContext) (model : TransformerChain<BinaryPredictionTransformer<FastForestClassificationModelParameters>>) description data = 
    let predictions = model.Transform data

    let metrics = ml.BinaryClassification.Evaluate(predictions)

    printfn ""
    printfn "### %s" description
    printfn "Accuracy          : %0.4f" (metrics.Accuracy)
    printfn "F1                : %0.4f" (metrics.F1Score)
    printfn "Positive Precision: %0.4f" (metrics.PositivePrecision)
    printfn "Positive Recall   : %0.4f" (metrics.PositiveRecall)
    printfn "Negative Precision: %0.4f" (metrics.NegativePrecision)
    printfn "Negative Recall   : %0.4f" (metrics.NegativeRecall)
    printfn ""

    let preview = predictions.Preview()
    
    preview.RowView
    |> Seq.take 5
    |> Seq.iter(fun row ->
        row.Values
        |> Array.iter (fun kv -> printfn "%s: %A" kv.Key kv.Value)
        printfn "")
    printfn ""

[<EntryPoint>]
let main argv =
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

    let predictor = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(context)

    let tests = 
        [
            "It was cool, cute, and funny."; 
            "It was very bad."; 
            "It was the greatest thing I've seen." 
        ]

    tests 
    |> List.iter (fun x ->
        let input = SentimentData()
        input.SentimentText <- x

        let prediction = predictor.Predict(input)
        printfn ""
        printfn "Text       : %s" x
        printfn "Prediction : %b" (prediction.PredictedLabel)
        printfn "Score      : %0.4f" (prediction.Score)
        )

    // Save model to file
    let saveModel (ml:MLContext) trainedModel = 
        use fsWrite = new FileStream("test-model.zip", FileMode.Create, FileAccess.Write, FileShare.Write)
        ml.Model.Save(trainedModel, fsWrite)

    saveModel context model

    // Load model from file
    use fsRead = new FileStream("test-model.zip", FileMode.Open, FileAccess.Read, FileShare.Read)
    let mlReloaded = MLContext()
    let modelReloaded = TransformerChain.LoadFrom(mlReloaded, fsRead)
    let predictorReloaded = modelReloaded.CreatePredictionEngine<SentimentData, SentimentPrediction>(mlReloaded)

    let test1 = SentimentData()
    test1.SentimentText <- tests.[0]

    let predictionReloaded = predictorReloaded.Predict(test1)
    printfn ""
    printfn "Text                  : %s" tests.[0]
    printfn "Prediction (Reloaded) : %b" (predictionReloaded.PredictedLabel)
    printfn "Score (Reloaded)      : %0.4f" (predictionReloaded.Score)

    0