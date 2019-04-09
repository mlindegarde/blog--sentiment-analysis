namespace Demo.SentimentAnalysis.Part1

open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Trainers.FastTree

module Models = 
    type SentimentData () =
        [<DefaultValue>]
        [<LoadColumn(0)>]
        val mutable public SentimentText :string

        [<DefaultValue>]
        [<LoadColumn(1)>]
        val mutable public Label :bool
        
        // NOTE: Need to add this column to extract metrics
        [<DefaultValue>]
        [<LoadColumn(2)>]
        val mutable public Probability :float32
       
    type SentimentPrediction () =
        [<DefaultValue>]
        val mutable public SentimentData :string

        [<DefaultValue>]
        val mutable public PredictedLabel :bool

        [<DefaultValue>]
        val mutable public Score :float32 

    type MyModel = TransformerChain<BinaryPredictionTransformer<FastForestClassificationModelParameters>>
    type MyPredictionEngine = PredictionEngine<SentimentData, SentimentPrediction>