namespace Demo.SentimentAnalysis.Part1

open Demo.SentimentAnalysis.Part1.Models
open Demo.SentimentAnalysis.Part1.Display

module Testing = 
    let testModel tests (predictor : MyPredictionEngine)  =
        tests 
        |> List.iter (
            fun text ->
                let input = SentimentData()

                // Generaly values in F# are immutable.  If you declare
                // something as mutable you can use the <- operator to
                // assign it a new value
                input.SentimentText <- text
            
                (input, predictor.Predict(input)) |> displayTestResult)