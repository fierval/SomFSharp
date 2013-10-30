using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Microsoft.FSharp.Core;
using FSom;

namespace SomTest
{
    [TestClass]
    public class EsomTests
    {
        [TestMethod]
        [TestCategory("PMatrix")]
        public void PariwiseDistanceTest()
        {
            List<Node> nodes = new List<Node>()
            {
                new Node(new double [] {0d, 0d, 2d}),
                new Node(new double [] {0d, 0d, 0d}),
                new Node(new double [] {0d, 0d, 0d})
            };

            var som = new Som(new Tuple<int, int>(3, 3), nodes);
            var dist = som.PairwiseDistance();

            Assert.AreEqual(2, dist[0 * 3 + 2 - (1 * 2 / 2)]);
            Assert.AreEqual(2, dist[0 * 3 + 1 - (1 * 2 / 2)]);
            Assert.AreEqual(0, dist[1 * 3 + 2 - (2 * 3 / 2)]);
        }

        [TestMethod]
        [TestCategory("PMatrix")]
        [DeploymentItem("proteomicsPBMC_mod.txt")]
        public void PariwiseDistancePerfTest()
        {
            var som = new Som(new Tuple<int, int>(3, 3), "proteomicsPBMC_mod.txt", new FSharpOption<int>(1));
            var dist = som.PairwiseDistance();
            Assert.IsNotNull(dist);
        }

        [TestMethod]
        [TestCategory("PMatrix")]
        [DeploymentItem("patents.txt")]
        public void ParetoRadiusTest()
        {
            var som = new Som(new Tuple<int, int>(3, 3), "patents.txt", new FSharpOption<int>(0));
            Assert.IsNotNull(som.ParetoRadius);
        }

        [TestMethod]
        [TestCategory("PMatrix")]
        [DeploymentItem("igrisk_v8_1.txt")]
        public void PMatrixTest()
        {
            var som = new SomGpuTest(new Tuple<int, int>(250, 250), "igrisk_v8_1.txt", new FSharpOption<int>(1));
            som.NormalizeInput(Normalization.Zscore);
            var density = som.DensityMatrix();
            Assert.IsNotNull(density);
        }

    }
}
