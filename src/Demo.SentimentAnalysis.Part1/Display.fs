namespace Demo.SentimentAnalysis.Part1

open Microsoft.ML
open Demo.SentimentAnalysis.Part1.Models

module Display = 
    let displayEvaluation (ml : MLContext) (model : MyModel) description data = 
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

    let displayTestResult (prediction : SentimentPrediction) =
        printfn ""
        printfn "Text       : %s" x
        printfn "Prediction : %b" (prediction.PredictedLabel)
        printfn "Score      : %0.4f" (prediction.Score)