using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public class Network
    {
        public Layer[] Layers { get; }

        public Network(params int[] layerSizes)
        {
            Layers = new Layer[layerSizes.Length - 1];
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i] = new Layer(layerSizes[i], layerSizes[i + 1]);
            }
        }

        public void LearnSlow(DataPoint[] trainingData, double learnRate)
        {
            // DO NOT USE - just to show the slow way of updating,
            // we have to calculate the cost for every single weight
            // and bias, which quickly becomes extremely slow
            throw new NotSupportedException($"No longer used or supported, " +
                $"use the {nameof(Learn)}() method");
            double h = 0.0001;
            double originalCost = Cost(trainingData);

            foreach (Layer layer in Layers)
            {
                // Calculate cost gradient
                Parallel.For(0, layer.NodesIn, nodeIn =>
                {
                    Parallel.For(0, layer.Nodes, node =>
                    {
                        layer.Weights[nodeIn, node] += h;
                        double deltaCost = Cost(trainingData) - originalCost;
                        layer.Weights[nodeIn, node] -= h;
                        layer.WeightGradients[nodeIn, node] = deltaCost / h;
                    });
                });

                // Calculate bias gradient
                Parallel.For(0, layer.Biases.Length, bias =>
                {
                    layer.Biases[bias] += h;
                    double deltaCost = Cost(trainingData) - originalCost;
                    layer.Biases[bias] -= h;
                    layer.BiasGradients[bias] = deltaCost / h;
                });

                layer.ApplyGradients(learnRate);
            }
        }

        public void Learn(DataPoint[] dataPoints, double learnRate)
        {
            // Update the gradients for each data point
            foreach (var dataPoint in dataPoints)
            {
                UpdateAllGradients(dataPoint);
            }

            // Average the gradients and apply
            ApplyAllGradients(learnRate / dataPoints.Length);

            // Clear all the gradients ready for next learning batch
            ClearAllGradients();
        }

        public void UpdateAllGradients(DataPoint dataPoint)
        {
            // Update gradients using backpropagation algorithm
            // Calculate gradients for nodes of last layer first
            CalculateOutputs(dataPoint.Inputs);
            double[] nodeValues = Layers.Last().CalculateOutputNodeValues(dataPoint.ExpectedOutputs);
            Layers.Last().UpdateGradients(nodeValues);

            // Then work backwards through all the other layers
            for (int i = Layers.Length - 2; i >= 0; i--)
            {
                var hiddenLayer = Layers[i];
                nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(Layers[i + 1], nodeValues);
                hiddenLayer.UpdateGradients(nodeValues);
            }
        }

        public double[] CalculateOutputs(double[] inputs)
        {
            foreach (var layer in Layers)
            {
                inputs = layer.CalculateOutputs(inputs);
            }
            return inputs;
        }

        public int Classify(double[] inputs)
        {
            var outputs = CalculateOutputs(inputs);
            return IndexOfMaxValue(outputs);
        }

        public double Cost(DataPoint dataPoint)
        {
            var outputs = CalculateOutputs(dataPoint.Inputs);
            return LayerCost(outputs, dataPoint.ExpectedOutputs);
        }

        public double Cost(DataPoint[] dataPoints)
            => dataPoints.Select(x => Cost(x)).Average();

        public static double LayerCost(double[] outputs, double[] expectedOutputs)
        {
            double cost = 0;
            for (int i = 0; i < outputs.Length; i++)
            {
                cost += Layer.NodeCost(outputs[i], expectedOutputs[i]);
            }
            return cost;
        }

        private int IndexOfMaxValue(IList<double> values)
        {
            return values.IndexOf(values.Max());
        }

        private void ApplyAllGradients(double learnRate)
        {
            Parallel.ForEach(Layers, layer => layer.ApplyGradients(learnRate));
        }

        private void ClearAllGradients()
        {
            foreach (var layer in Layers)
            {
                layer.ClearGradients();
            }
        }
    }
}
