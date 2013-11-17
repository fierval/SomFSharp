using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using FSom;
using Microsoft.FSharp.Core;

namespace SomTest
{
    [TestClass]
    public class ClusterTest
    {
        [TestMethod]
        [TestCategory("Clustering")]
        [DeploymentItem("res_dense.txt")]
        [DeploymentItem("res_dist.txt")]
        public void AscentDescentTest()
        {
            var dense = Som.ReadMatrix("res_dense.txt", new Func<string, int>(int.Parse));
            var dist = Som.ReadMatrix("res_dist.txt", new Func<string, double>(double.Parse));

            var ascentDescent = new Clustering.AscentDescent(dist, dense);
            ascentDescent.Immerse();
            Assert.IsNotNull(ascentDescent.Immersion);

            var classes = Clustering.AscentDescent.ImmersionToClasses(ascentDescent.Immersion);
            Assert.IsNotNull(classes);
        }

        [TestMethod]
        [TestCategory("Clustering")]
        [DeploymentItem("res_ustar.txt")]
        public void WatershedTest()
        {
            var ustar = Som.ReadMatrix("res_ustar.txt", new Func<string, double>(double.Parse));

            var waterhsed = new Clustering.Watershed(ustar);
            waterhsed.CreateWatersheds();
            Assert.IsNotNull(waterhsed.Labels);
        }

        [TestMethod]
        [TestCategory("Clustering")]
        [DeploymentItem("proteomicsPBMC_som_dense.txt")]
        [DeploymentItem("proteomicsPBMC_som_dist.txt")]
        [DeploymentItem("proteomicsPBMC_som_ustar.txt")]
        [DeploymentItem("proteomicsPBMC_mod.txt")]
        public void ClassifyImmersionTest()
        {
            var som = new Som(new Tuple<int, int>(150, 150), "proteomicsPBMC_mod.txt", new FSharpOption<int>(1));

            var dense = Som.ReadMatrix("proteomicsPBMC_som_dense.txt", new Func<string, int>(int.Parse));
            var dist = Som.ReadMatrix("proteomicsPBMC_som_dist.txt", new Func<string, double>(double.Parse));

            var ustar = Som.ReadMatrix("proteomicsPBMC_som_ustar.txt", new Func<string, double>(double.Parse));

            var clusters = som.Cluster(dist, dense, ustar);

            List<string> output = new List<string>();
            for (int i = 0; i < som.Height; i++)
            {
                string row = String.Empty;
                for (int j = 0; j < som.Width; j++)
                {
                    row += "\t";
                    row += clusters[i,j].ToString();
                }
                output.Add(row.Substring(1));
            }
            File.WriteAllLines(@"C:\Users\boris\Dropbox\Algorithms\Visualizations\som\data\proteomicsPBMC_som_watershed.txt", output);
        }

        [TestMethod]
        [TestCategory("Clustering")]
        [DeploymentItem("patents1.txt")]
        public void ClusteringTest()
        {
            var som = new Som(new Tuple<int, int>(10, 20), "patents1.txt", FSharpOption<int>.None);
            som.NormalizeInput(Normalization.Zscore);

            som.Train(100);

            var distance = som.DistanceMap();
            var density = som.DensityMatrix();
            var umatrix = som.UStarMatrix(distance, density);

            var cluster = som.Cluster(distance, density, umatrix);

            Assert.IsNotNull(cluster);
        }
    }
}
