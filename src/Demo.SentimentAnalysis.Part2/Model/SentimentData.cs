namespace Demo.SentimentAnalysis.Part2.Model
{
    public class SentimentData
    {
        #region Properties
        public string SentimentText { get; set; }
        public bool Label { get; set; }
        public float Probability { get; set; }
        #endregion
    }
}
