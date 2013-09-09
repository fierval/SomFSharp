using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using FSom;
using System.Collections.Generic;
using System.Linq;

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
    }
}
