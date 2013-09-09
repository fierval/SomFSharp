﻿namespace FSom

open System
open System.Collections
open System.Collections.Generic
open System.Linq
open MathNet.Numerics
open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra.Generic
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Random
open MathNet.Numerics.Distributions

type Node(weights : float seq) =
    let weights = DenseVector.OfEnumerable(weights)
    static let rndSource = new MersenneTwister()
    static let gamma = Gamma(2.0, 1.5)

    do
        match weights.Count with
        | 0 -> failwith "empty node"
        | _ -> ignore()

    static do
        gamma.RandomSource <- rndSource

    new (len) = Node(gamma.Samples().Take(len))
        
    member this.Item 
        with get(index) = weights.[index]

    member this.Dimension = weights.Count

    /// Enumeration interfaces to enable
    /// treatment of node as a sequence of its values
    interface IEnumerable with
        member this.GetEnumerator() = weights.GetEnumerator() :> IEnumerator

    interface IEnumerable<float> with
        member this.GetEnumerator() = weights.GetEnumerator()

