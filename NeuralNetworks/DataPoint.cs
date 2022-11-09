using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    public struct DataPoint
    {
        public double[] Inputs { get; set; }
        public double[] ExpectedOutputs { get; set; }
    }
}
