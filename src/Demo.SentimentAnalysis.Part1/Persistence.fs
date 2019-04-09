namespace Demo.SentimentAnalysis.Part1

open System.IO
open Microsoft.ML
open Microsoft.ML.Data

open Demo.SentimentAnalysis.Part1.Models

module Persistence =
    let saveModel fileName (context : MLContext, model : MyModel) = 
        use fileStream = new FileStream(fileName, FileMode.Create, FileAccess.Write, FileShare.Write)
        context.Model.Save(model, fileStream)

        (context, model)

    let loadModel fileName =
        use fileStream = new FileStream(fileName, FileMode.Open, FileAccess.Read, FileShare.Read)
        let context = MLContext()
        let model = TransformerChain.LoadFrom(context, fileStream)

        (context, model)