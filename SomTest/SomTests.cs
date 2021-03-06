﻿using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using FSom;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Random;
using System.Threading.Tasks;
using Microsoft.FSharp.Core;
using System.IO;
using Microsoft.FSharp.Collections;

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

            var range = Enumerable.Range(1, 120);
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
            som.train(100, FSharpOption<bool>.None);
            var map = som.somMap;
        }

        [TestMethod]
        [TestCategory("Parallel")]
        public void TrainingSeqTest()
        {
            var gamma = new Gamma(2.0, 1.5);
            gamma.RandomSource = new MersenneTwister();

            var range = Enumerable.Range(1, 100).ToList();
            var nodes = range.AsParallel().Select(r => new Node(gamma.Samples().Take(12))).ToList();

            var som = new Som(new Tuple<int, int>(200, 200), nodes);
            som.train(2, FSharpOption<bool>.Some(false));
        }

        [TestMethod]
        [TestCategory("Parallel")]
        public void TrainingParallelTest()
        {
            var gamma = new Gamma(2.0, 1.5);
            gamma.RandomSource = new MersenneTwister();

            var range = Enumerable.Range(1, 100).ToList();
            var nodes = range.AsParallel().Select(r => new Node(gamma.Samples().Take(12))).ToList();

            var som = new Som(new Tuple<int, int>(200, 200), nodes);
            som.train(2, FSharpOption<bool>.Some(true));
        }

        [TestMethod]
        [TestCategory("Som")]
        public void BmuGpuSepParallelCompareTest()
        {
            var bound = 12;
            List<Node> nodes = Enumerable.Range(1, bound).Select(r => new Node(12)).ToList();
            var som = new SomGpuTest(new Tuple<int, int>(200, 200), nodes);

            var rnd = new Random((int)DateTime.Now.Ticks);
            int ind = rnd.Next(0, bound);

            Tuple<int, int> ij = som.GetBMU(nodes[ind]);
            var i = ij.Item1;
            var j = ij.Item2;

            var gpuMins = som.GetBmuGpuSingle(nodes);
            var ijG = som.toSomCoordinates(gpuMins[ind]);

            Assert.AreEqual(i, ijG.Item1);
            Assert.AreEqual(j, ijG.Item2);
        }

        [TestMethod]
        [TestCategory("Som")]
        public void TrainGpuTest()
        {
            var bound = 12;
            List<Node> nodes = Enumerable.Range(1, bound).Select(r => new Node(3)).ToList();
            var som = new SomGpuTest(new Tuple<int, int>(5, 5), nodes);

            var rnd = new Random((int)DateTime.Now.Ticks);
            int ind = rnd.Next(0, bound);

            Tuple<int, int> ij = som.GetBMU(nodes[ind]);
            var i = ij.Item1;
            var j = ij.Item2;

            som.SingleDimTrain(nodes[0]);
        }

        [TestMethod]
        [TestCategory("Som")]
        [DeploymentItem("patents.txt")]
        public void DistMapSingleTest()
        {

            var som = new SomGpuTest(new Tuple<int, int>(10, 20), "patents.txt", FSharpOption<int>.None);
            var dist = som.GetDistanceMapSingle();

            var distExp = som.LinearGetDistanceMap();
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 20; j++)
                {
                    Assert.AreEqual((decimal)distExp[i, j], (decimal)dist[i, j]);
                }
            }
        }
        
        [TestMethod]
        [TestCategory("IO")]
        [DeploymentItem("patents.txt")]
        public void ReadFormat0Test()
        {
            var som = new SomGpu(new Tuple<int, int>(3, 3), "patents.txt", FSharpOption<int>.None);
            Assert.AreEqual(true, som.ShouldClassify);
            Assert.AreEqual(4, som.NodeLen);
            Assert.AreEqual("1", som.InputNodes[0].Class);
            Assert.AreEqual("Apple", som.InputNodes[7].Name);
            Assert.AreEqual("7", som.InputNodes[7].Class);
        }

        [TestMethod]
        [TestCategory("IO")]
        [DeploymentItem("patents1.txt")]
        public void ReadFormat1Test()
        {
            var som = new SomGpu(new Tuple<int, int>(3, 3), "patents1.txt", FSharpOption<int>.None);
            Assert.AreEqual(true, som.ShouldClassify);
            Assert.AreEqual(4, som.NodeLen);
            Assert.AreEqual("1", som.InputNodes[0].Class);
            Assert.AreEqual("Apple", som.InputNodes[7].Name);
            Assert.AreEqual("7", som.InputNodes[7].Class);
        }

        [TestMethod]
        [TestCategory("IO")]
        [DeploymentItem("patents1.txt")]
        public void SavingTest()
        {
            var som = new Som(new Tuple<int, int>(10, 20), "patents1.txt", FSharpOption<int>.None);
            som.NormalizeInput(Normalization.Zscore);
            som.Train(20);
            if (!Directory.Exists(@"c:\temp"))
            {
                Directory.CreateDirectory(@"c:\temp");
            }
            som.Save(20, @"c:\temp\res.txt");
        }


        [TestMethod]
        [TestCategory("Som")]
        public void BmuGpuShortMapTest()
        {
            var bound = 1200;
            int dim1 = 5;
            int dim2 = 5;
            List<Node> nodes = Enumerable.Range(1, bound).Select(r => new Node(12)).ToList();
            var som = new SomGpuTest(new Tuple<int, int>(dim1, dim2), nodes);

            var rnd = new Random((int)DateTime.Now.Ticks);
            int ind = rnd.Next(0, bound);

            Tuple<int, int> ij = som.GetBMUParallel(nodes[ind]);
            var i = ij.Item1;
            var j = ij.Item2;

            var gpuMins = som.GetBmuGpuShortMapSingle(som.asArray, nodes);
            var ijG = som.toSomCoordinates(gpuMins[ind]);

            Assert.AreEqual(i, ijG.Item1);
            Assert.AreEqual(j, ijG.Item2);
        }

        [TestMethod]
        [TestCategory("Som")]
        [DeploymentItem("patents1.txt")]
        public void GetBmuUnifiedTest()
        {
            var som = new SomGpuTest(new Tuple<int, int>(10, 30), "patents1.txt", FSharpOption<int>.None);
            som.NormalizeInput(Normalization.Zscore);
            var bmus = som.GetBmuGpuShortMapSingle(som.InputNodes.SelectMany(n => n.AsEnumerable()).ToArray(), SeqModule.Cast<Node>(som.somMap));
            Assert.IsNotNull(bmus);
        }

    }
}
