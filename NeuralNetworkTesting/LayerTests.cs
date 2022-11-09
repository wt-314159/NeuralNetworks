namespace NeuralNetworkTesting
{
    [TestClass]
    public class LayerTests
    {
        [TestMethod]
        public void TestCalculateOutput()
        {
            var layer = new Layer(2, 2);
            layer.Weights = new double[2, 2] { { 1, 2 }, { 2, 1 } };
            layer.Biases = new double[] { 1, 2 };

            var outputs = layer.CalculateOutputs(new double[] { 2, 2 });
        }
    }
}