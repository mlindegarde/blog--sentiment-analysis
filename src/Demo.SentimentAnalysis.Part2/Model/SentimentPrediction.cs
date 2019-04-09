namespace Demo.SentimentAnalysis.Part2.Model
{
    public class SentimentPrediction
    {
        #region Properties
        public string SentimentData { get; set; }
        public bool PredictedLabel { get; set; }
        public float Score { get; set; }
        #endregion
    }
}
