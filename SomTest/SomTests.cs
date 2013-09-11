﻿using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using FSom;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Random;
using System.Threading.Tasks;

namespace SomTest
{
    [TestClass]
    public class SomTests
    {
        [TestMethod]
        [TestCategory("Som")]
        public void InitSomTest()
        {
            List<Node> nodes = new List<Node>()
            {
                new Node(10),
                new Node(10),
                new Node(10)
            };

            var som = new Som(new Tuple<int,int>(5, 5), nodes);
            Assert.IsNotNull(som);
        }

        [TestMethod]
        [TestCategory("Som")]
        public void BmuSeqParallelCompareTest()
        {
            List<Node> nodes = new List<Node>()
            {
                new Node(5),
                new Node(5),
                new Node(5)
            };

            var som = new Som(new Tuple<int, int>(20, 20), nodes);

            Tuple<int, int> ij = som.GetBMU(nodes[0]);
            var i = ij.Item1;
            var j = ij.Item2;

            ij = som.GetBMUParallel(nodes[0]);
            var iP = ij.Item1;
            var jP = ij.Item2;

            Assert.AreEqual(i, iP);
            Assert.AreEqual(j, jP);
        }

        [TestMethod]
        [TestCategory("Parallel")]
        public void BmuParallelTest()
        {
            var gamma = new Gamma(2.0, 1.5);
            gamma.RandomSource = new MersenneTwister();

            var range = Enumerable.Range(1, 1);
            var nodes = range.AsParallel().Select(r => new Node(gamma.Samples().Take(12))).ToList();

            var som = new Som(new Tuple<int, int>(200, 200), nodes);

            var ij = som.GetBMUParallel(nodes.First());
            var iP = ij.Item1;
            var jP = ij.Item2;

        }

        [TestMethod]
        [TestCategory("Parallel")]
        public void BmuSeqTest()
        {
            var gamma = new Gamma(2.0, 1.5);
            gamma.RandomSource = new MersenneTwister();

            var range = Enumerable.Range(1, 1);
            var nodes = range.AsParallel().Select(r => new Node(gamma.Samples().Take(12))).ToList();

            var som = new Som(new Tuple<int, int>(200, 200), nodes);

            Tuple<int, int> ij = som.GetBMU(nodes[0]);
            var i = ij.Item1;
            var j = ij.Item2;
        }

        [TestMethod]
        [TestCategory("Parallel")]
        public void BmuParallelLargeSeqTest()
        {
            var gamma = new Gamma(2.0, 1.5);
            gamma.RandomSource = new MersenneTwister();

            var range = Enumerable.Range(1, 100).ToList();
            var nodes = range.AsParallel().Select(r => new Node(gamma.Samples().Take(12))).ToList();

            var som = new Som(new Tuple<int, int>(200, 200), nodes);

            for (int i = 0; i < range.Count; i++)
            {
                Tuple<int, int> ij = som.GetBMUParallel(nodes[0]);

                Assert.IsNotNull(ij);
            }
        }

        [TestMethod]
        [TestCategory("Parallel")]
        public void BmuSeqLargeParallelTest()
        {
            var gamma = new Gamma(2.0, 1.5);
            gamma.RandomSource = new MersenneTwister();

            var range = Enumerable.Range(1, 100).ToList();
            var nodes = range.AsParallel().Select(r => new Node(gamma.Samples().Take(12))).ToList();

            var som = new Som(new Tuple<int, int>(200, 200), nodes);

            Parallel.For(0, range.Count, (i) =>
            {
                Tuple<int, int> ij = som.GetBMU(nodes[0]);

                Assert.IsNotNull(ij);
            });
        }

        [TestMethod]
        [TestCategory("Parallel")]
        public void BmuParallelLargeParallelTest()
        {
            var gamma = new Gamma(2.0, 1.5);
            gamma.RandomSource = new MersenneTwister();

            var range = Enumerable.Range(1, 100).ToList();
            var nodes = range.AsParallel().Select(r => new Node(gamma.Samples().Take(12))).ToList();

            var som = new Som(new Tuple<int, int>(200, 200), nodes);

            Parallel.For(0, range.Count, (i) =>
            {
                Tuple<int, int> ij = som.GetBMU(nodes[0]);

                Assert.IsNotNull(ij);
            });
        }

        [TestMethod]
        [TestCategory("Som")]
        public void TrainingTest()
        {
            List<Node> nodes = new List<Node>()
            {
                new Node(new double [] {0d, 0d, 255d}),
                new Node(new double [] {0d, 255d, 0d}),
                new Node(new double [] {255d, 0d, 0d})
            };

            var som = new Som(new Tuple<int, int>(6, 6), nodes);
            var untrained = som.somMap;
            som.train(100);
            var map = som.somMap;
        }
    }
}
