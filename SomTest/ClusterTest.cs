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

    }
}
