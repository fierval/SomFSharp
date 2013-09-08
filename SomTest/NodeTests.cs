using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using FSom;

namespace SomTest
{
    [TestClass]
    public class NodeTests
    {
        [TestMethod]
        [TestCategory("Node Tests")]
        public void CreateNodeTest()
        {
            Node node = new Node(new double[] { 25d, 313d, 1d, 0d });
            Assert.AreEqual(313d, node[1]);
        }

        [TestMethod]
        [TestCategory("Node Tests")]
        public void CreateRandomNodeTest()
        {
            Node node = new Node(25);
            Assert.AreEqual(25, node.Dimension);
        }
    }
}
