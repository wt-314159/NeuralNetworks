using System.Xml.Linq;

namespace NeuralNetworks
{
    public class Layer
    {
        private double[] _inputs;
        private double[] _activations;
        private double[] _weightedOutputs;
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
            _inputs = inputs;
            _activations = new double[Nodes];
            _weightedOutputs = new double[Nodes];
            Parallel.For(0, Nodes, node =>
            {
                _weightedOutputs[node] = Biases[node];
                Parallel.For(0, NodesIn, nodeIn =>
                {
                    _weightedOutputs[node] += inputs[nodeIn] * Weights[nodeIn, node];
                });
                _activations[node] = SigmoidActivationFunction(_weightedOutputs[node]);
            });
            return _activations;
        }

        public double ActivationFunction(double input)
            => input > 0 ? 1 : 0;

        public double SigmoidActivationFunction(double input)
            => 1 / (1 + Math.Exp(-input));

        public double SigmoidActivationDerivative(double input)
        {
            var activation = SigmoidActivationFunction(input);
            return activation * (1 - activation);
        }

        public static double NodeCost(double output, double expectedValue)
        {
            var error = output - expectedValue;
            return error * error;
        }

        public static double NodeCostDerivative(double output, double expectedValue)
            => 2 * output - expectedValue;

        public double[] CalculateOutputNodeValues(double[] expectedValues)
        {
            var nodeValues = new double[expectedValues.Length];

            Parallel.For(0, expectedValues.Length, i =>
            {
                var costDerivative = NodeCostDerivative(_activations[i], expectedValues[i]);
                var activationDerivative = SigmoidActivationDerivative(_weightedOutputs[i]);
                nodeValues[i] = activationDerivative * costDerivative;
            });

            return nodeValues;
        }

        public double[] CalculateHiddenLayerNodeValues(Layer nextLayer, double[] nextLayerNodeValues)
        {
            var nodeValues = new double[Nodes];
            for (int newNodeIndex = 0; newNodeIndex < nodeValues.Length; newNodeIndex++)
            {
                double value = 0;
                for (int oldNodeIndex = 0; oldNodeIndex < nextLayerNodeValues.Length; oldNodeIndex++)
                {
                    double weightedInputDerivative = nextLayer.Weights[newNodeIndex, oldNodeIndex];
                    value += weightedInputDerivative * nextLayerNodeValues[oldNodeIndex];
                }
                value *= SigmoidActivationDerivative(_inputs[newNodeIndex]);
                nodeValues[newNodeIndex] = value;
            }
            return nodeValues;
        }

        public void UpdateGradients(double[] nodeValues)
        {
            for (int node = 0; node < Nodes; node++)
            {
                for (int nodeIn = 0; nodeIn < NodesIn; nodeIn++)
                {
                    double derivativeCostWeight = _inputs[nodeIn] * nodeValues[node];
                    WeightGradients[nodeIn, node] += derivativeCostWeight;
                }
                BiasGradients[node] += nodeValues[node];
            }
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

        public void ClearGradients()
        {
            WeightGradients = new double[NodesIn, Nodes];
            BiasGradients = new double[Nodes];
        }
    }
}