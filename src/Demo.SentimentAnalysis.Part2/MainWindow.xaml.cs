using System.Globalization;
using System.IO;
using System.Windows;
using Demo.SentimentAnalysis.Part2.Model;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Demo.SentimentAnalysis.Part2
{
    public partial class MainWindow : Window
    {
        private PredictionEngine<SentimentData, SentimentPrediction> _predictor;

        public MainWindow()
        {
            InitializeComponent();
            LoadModel();
        }

        private void LoadModel()
        {
            // Here the model is being loaded from a file.  We could also embed the model in an
            // assembly as a resource.  This would then allow us to update the model via a NuGet
            // package.
            using(FileStream fileStream = new FileStream("test-model.zip", FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                // The MLContext is the starting point for all ML things using ML.Net.
                MLContext context = new MLContext();

                // Build the model from the data contained within the zip file.
                TransformerChain<ITransformer> model = TransformerChain.LoadFrom(context, fileStream);

                // Create the predictor we'll use whenever the user clicks a button.
                _predictor = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(context);
            }
        }

        private void OnDoIt(object sender, RoutedEventArgs e)
        {
            // Set the output to the score returned from the predictor we created during the
            // creation of the window.
            Output.Text = 
                _predictor
                    .Predict(
                        new SentimentData
                        {
                            SentimentText = Input.Text
                        })
                    .Score.ToString(CultureInfo.InvariantCulture);
        }
    }
}
