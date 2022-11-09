namespace NeuralNetworks
{
    public class Layer
    {
        public int NodesIn { get; }
        public int Nodes { get; }
        public double[,] Weights { get; set; }
        public double[] Biases { get; set; }

        public Layer(int nodesIn, int nodes)
        {
            NodesIn = nodesIn;
            Nodes = nodes;

            Weights = new double[nodesIn, nodes];
            Biases = new double[nodes];
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
    }
}