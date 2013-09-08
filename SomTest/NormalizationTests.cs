using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using System.Linq;
using FSom;
using Microsoft.FSharp.Collections;
using MathNet.Numerics.Statistics;
using MathNet.Numerics.Random;
using MathNet.Numerics.Distributions;

namespace SomTest
{
    [TestClass]
    public class NormalizationTests
    {
        [TestMethod]
        [TestCategory("Normalization Constants")]
        public void MinMaxConstTest()
        {
            List<IList<double>> seqs = new List<IList<double>>
            {
                new List<double>() { 5d, 7d, 1d, 1.8d},
                new List<double>() { 1.5d, 27d, 3d, 2.1d},
                new List<double>() { 0.02d, 3.23d, 11d, 1.8d }
            };

            var tpl = normalize.getNormalConstants(seqs, Normalization.MinMax);

            Assert.AreEqual(-.02d, tpl[0].Item1);
            Assert.AreEqual(0.9d / (27d - 3.23d), tpl[1].Item2);
        }

        [TestMethod]
        [TestCategory("Normalization Constants")]
        public void ZscoreTest()
        {
            List<IList<double>> seqs = new List<IList<double>>
            {
                new List<double>() { 5d, 7d, 1d, 1.8d},
                new List<double>() { 1.5d, 27d, 3d, 2.1d},
                new List<double>() { 0.02d, 3.23d, 11d, 1.8d }
            };

            var tpl = normalize.getNormalConstants(seqs, Normalization.Zscore);

            Assert.AreEqual(-(new double [] {5d, 1.5d, 0.02d}).Average(), tpl[0].Item1);
            Assert.AreEqual(0.9 / ((new double [] {7d, 27d, 3.23d}).StandardDeviation()), tpl[1].Item2);
        }

        [TestMethod]
        [TestCategory("Normalization Constants")]
        public void NormalizeConstBulkTest()
        {
            var gamma = new Gamma(2.0, 1.5);
            gamma.RandomSource = new MersenneTwister();

            var range = Enumerable.Range(1, 100000);
            IList<IList<double>> seqs = range.AsParallel().Select(r => gamma.Samples().Take(30).ToList() as IList<double>).ToList();

            var tpl = normalize.getNormalConstants(seqs, Normalization.Zscore);
            Assert.IsNotNull(tpl);
        }

        [TestMethod]
        [TestCategory("Normalization")]
        public void NormalizeTest()
        {
            List<IList<double>> seqs = new List<IList<double>>
            {
                new List<double>() { 5d, 7d, 1d, 1.8d},
                new List<double>() { 1.5d, 27d, 3d, 2.1d},
                new List<double>() { 0.02d, 3.23d, 11d, 1.8d }
            };

            var normalized = normalize.normalize(seqs, Normalization.Zscore).ToList();
            
            var add = -(new double[] { 7d, 27d, 3.23d }).Average();
            var mult = 0.9 / (new double[] { 7d, 27d, 3.23d }).StandardDeviation();
            var expected = (7d + add) * mult;

            Assert.AreEqual(expected, normalized[0].Take(2).Last());
        }

        [TestMethod]
        [TestCategory("Normalization")]
        public void NormalizeBulkTest()
        {
            var gamma = new Gamma(2.0, 1.5);
            gamma.RandomSource = new MersenneTwister();

            var range = Enumerable.Range(1, 100000);
            IList<IList<double>> seqs = range.AsParallel().Select(r => gamma.Samples().Take(30).ToList() as IList<double>).ToList();

            var normalized = normalize.normalize(seqs, Normalization.Zscore).AsParallel().Select(x => x.ToList()).ToList();

        }

    }
}
