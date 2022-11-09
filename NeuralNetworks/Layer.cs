namespace NeuralNetworks
{
    public class Layer
    {
        public int NodesIn { get; }
        public int Nodes { get; }
        public double[,] Weights { get; set; }
        public double[] Biases { get; set; }
        public double[,] WeightGradients { get; set; }
        public double[] BiasGradients { get; set; }

        public Layer(int nodesIn, int nodes)
        {
            NodesIn = nodesIn;
            Nodes = nodes;

            Weights = new double[nodesIn, nodes];
            Biases = new double[nodes];

            WeightGradients = new double[nodesIn, nodes];
            BiasGradients = new double[nodes];

            RadomiseWeights();
        }

        public void RadomiseWeights()
        {
            var rand = new Random();
            var sqrt = Math.Sqrt(NodesIn);

            for (int node = 0; node < Nodes; node++)
            {
                for (int nodeIn = 0; nodeIn < NodesIn; nodeIn++)
                {
                    var randomDouble = rand.NextDouble() * 2 - 1;       // random num between -1 and 1
                    Weights[nodeIn, node] = randomDouble / sqrt;        // normalize by square root of 
                                                                        // number of inputs
                }
            }
        }

        public double[] CalculateOutputs(double[] inputs)
        {
            var activations = new double[Nodes];
            Parallel.For(0, Nodes, node =>
            {
                var output = Biases[node];
                Parallel.For(0, NodesIn, nodeIn =>
                {
                    output += inputs[nodeIn] * Weights[nodeIn, node];
                });
                activations[node] = SigmoidActivationFunction(output);
            });
            return activations;
        }

        public double ActivationFunction(double activation)
            => activation > 0 ? 1 : 0;

        public double SigmoidActivationFunction(double activation)
            => 1 / (1 + Math.Exp(-activation));

        public static double NodeCost(double output, double expectedValue)
        {
            var error = expectedValue - output;
            return error * error;
        }

        public void ApplyGradients(double learnRate)
        {
            for (int node = 0; node < Nodes; node++)
            {
                Biases[node] -= BiasGradients[node] * learnRate;
                for (int nodeIn = 0; nodeIn < NodesIn; nodeIn++)
                {
                    Weights[nodeIn, node] -= WeightGradients[nodeIn, node] * learnRate;
                }
            }
        }
    }
}