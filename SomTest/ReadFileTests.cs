using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using FSom;
using System.Collections.Generic;
using System.Linq;

namespace SomTest
{
    [TestClass]
    public class ReadFileTests
    {
        [TestMethod]
        [TestCategory("Read")]
        public void ReadTest()
        {
            string fileName = @"C:\Users\boris\Dropbox\Algorithms\Visualizations\som\som_src\src\Debug\patents.txt";
            var nodes = Utils.read(fileName).ToList();
            Assert.AreEqual(10, nodes.Count());
            Assert.AreEqual(6.05502567865004, nodes[0].Item3[1]);
            Assert.AreEqual(1770, nodes[9].Item3[3]);
        }

        [TestMethod]
        [TestCategory("Read")]
        public void ReadNormalizeTest()
        {
            string fileName = @"C:\Users\boris\Dropbox\Algorithms\Visualizations\som\som_src\src\Debug\patents.txt";
            var nodes = Utils.readAndNormalize(fileName, Normalization.MinMax).ToList();
            Assert.AreEqual(10, nodes.Count);
            Assert.AreEqual(0d, nodes[1].Item3[1]);
        }
    }
}
