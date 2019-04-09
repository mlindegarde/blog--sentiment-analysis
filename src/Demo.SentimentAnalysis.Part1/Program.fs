open System
open Microsoft.ML
open Microsoft.ML.Data

open Demo.SentimentAnalysis.Part1.Models
open Demo.SentimentAnalysis.Part1.Persistence
open Demo.SentimentAnalysis.Part1.Training
open Demo.SentimentAnalysis.Part1.Testing

let createPredictorFromMyModel (context : MLContext, model : MyModel) =
    model.CreatePredictionEngine<SentimentData, SentimentPrediction>(context)

let createPredictorFromLoadedModel (context : MLContext, model : TransformerChain<ITransformer>) =
    model.CreatePredictionEngine<SentimentData, SentimentPrediction>(context)

[<EntryPoint>]
let main argv =
    // function composition
    let saveModelAndCreatePredictor = (saveModel "test-model.zip") >> createPredictorFromMyModel

    let tests = 
        [
            "It was cool, cute, and funny."; 
            "It was very bad."; 
            "It was the greatest thing I've seen." 
        ]

    // partial function application
    let runTests = tests |> testModel

    // training the model
    printfn "Testing trained model"
    trainModel 
    |> saveModelAndCreatePredictor
    |> runTests
    printfn ""

    // reloading the mdoel and running the tests again
    printfn "Testing loaded model"
    loadModel "test-model.zip"
    |> createPredictorFromLoadedModel 
    |> runTests

    Console.ReadLine() |> ignore
    0