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
            => dataPoints.Select(x => Cost(x)).Sum();

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
    }
}
