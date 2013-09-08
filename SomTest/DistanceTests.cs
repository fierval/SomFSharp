using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using FSom;

namespace SomTest
{
    [TestClass]
    public class DistanceTests
    {
        [TestMethod]
        [TestCategory("Distance")]
        public void TestEuclDistance()
        {
            Node node1 = new Node(new double[] { 8d, 3d, 8d });
            Node node2 = new Node(new double[] { 4d, 2d, 0d });

            var dist = distance.getDistance(node1, node2, Metric.Euclidian);
            Assert.AreEqual(9d, dist);
        }

        [TestMethod]
        [TestCategory("Distance")]
        public void TestTaxicabDistance()
        {
            Node node1 = new Node(new double[] { 8d, 3d, 8d });
            Node node2 = new Node(new double[] { 4d, 2d, 0d });

            var dist = distance.getDistance(node1, node2, Metric.Taxicab);
            Assert.AreEqual(13d, dist);
        }
    }
}
