﻿using System;
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

            var somGpu = new SomGpuModule.SomGpu(new Tuple<int, int>(6, 6), nodes);
            Assert.AreEqual(somGpu[0, 5][1], somGpu.toArray[0 * 6 * 3 + 5 * 3 + 1]);
            Assert.AreEqual(somGpu[2, 4][0], somGpu.toArray[2 * 6 * 3 + 4 * 3 + 0]);
        }

        [TestMethod]
        [TestCategory("SomGpu")]
        public void GetSingleBmuTest()
        {
            List<Node> nodes = new List<Node>()
            {
                new Node(new double [] {0d, 0d, 255d}),
                new Node(new double [] {0d, 255d, 0d}),
                new Node(new double [] {255d, 0d, 0d})
            };

            var somGpu = new SomGpuModule.SomGpu(new Tuple<int, int>(6, 6), nodes);
            var bmu = somGpu.GetBMU(nodes[0]);
            var x = bmu.Item1;
            var y = bmu.Item2;

            var mins = somGpu.SingleDimBmu(100, nodes[0]);
            var min = mins.Min();
            var j = -1;
            mins.Where((e, i) => { if (e == min) j = i; return e == min; });
        }

    }
}
