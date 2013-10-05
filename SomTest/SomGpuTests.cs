using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using FSom;

namespace SomTest
{
    [TestClass]
    public class SomGpuTests
    {
        [TestMethod]
        [TestCategory("SomGpu")]
        public void ToArrayTest()
        {
            List<Node> nodes = new List<Node>()
            {
                new Node(new double [] {0d, 0d, 255d}),
                new Node(new double [] {0d, 255d, 0d}),
                new Node(new double [] {255d, 0d, 0d})
            };

            var somGpu = new SomGpuModule.SomGpu(new Tuple<int, int>(10, 6), nodes);
            Assert.AreEqual(somGpu[0, 5][1], somGpu.asArray[0 * 6 * 3 + 5 * 3 + 1]);
            Assert.AreEqual(somGpu[2, 4][0], somGpu.asArray[2 * 6 * 3 + 4 * 3 + 0]);
        }

        [TestMethod]
        [TestCategory("SomGpu")]
        public void ToSomCoordinatesTest()
        {
            List<Node> nodes = new List<Node>()
            {
                new Node(new double [] {0d, 0d, 255d}),
                new Node(new double [] {0d, 255d, 0d}),
                new Node(new double [] {255d, 0d, 0d})
            };

            var somGpu = new SomGpuModule.SomGpu(new Tuple<int, int>(6, 6), nodes);
            Assert.AreEqual(2, somGpu.toSomCoordinates(13).Item1);
            Assert.AreEqual(1, somGpu.toSomCoordinates(13).Item2);
        }

        [TestMethod]
        [TestCategory("SomGpu")]
        public void MergeNodesGpuTest()
        {
            List<Node> nodes = new List<Node>()
            {
                new Node(new double [] {0d, 0d, 255d}),
                new Node(new double [] {0d, 255d, 0d}),
                new Node(new double [] {255d, 0d, 0d})
            };

            var somGpu = new SomGpuModule.SomGpu(new Tuple<int, int>(6, 6), nodes);
            var flat = somGpu.MergeNodes();
        }

    }
}
