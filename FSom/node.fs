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

open System.Diagnostics

[<DebuggerDisplay("{Display}")>]
type Node(weights : float seq) as this =
    let weights = DenseVector.OfEnumerable(weights)
    [<DefaultValue>] val mutable private name : string
    [<DefaultValue>] val mutable private classs : string

    static let rndSource = new MersenneTwister(Random.timeSeed())

    do
        match weights.Count with
        | 0 -> failwith "empty node"
        | _ -> ignore()

        this.classs <- String.Empty
        this.name <- String.Empty

    new (len) = Node(seq {for i in [1..len] -> rndSource.NextDouble()})
    new (name, classs, (weights : float seq)) as this = 
        Node(weights)
        then
            this.name <- name
            this.classs <- classs
    
    member this.Display = weights.Aggregate("{", (fun a e -> a + e.ToString() + ", "), (fun a -> a.Substring(0, a.Length - 2) + "}"))
    member this.Name 
        with get () = this.name
        and set value = this.name <- value
            
    member this.Item 
        with get(index) = weights.[index]
        and set index value = weights.[index] <- value

    member this.Dimension = weights.Count
    member this.Class 
        with get () = this.classs
        and set value = this.classs <- value

    /// Enumeration interfaces to enable
    /// treatment of node as a sequence of its values
    interface IEnumerable with
        member this.GetEnumerator() = weights.GetEnumerator() :> IEnumerator

    interface IEnumerable<float> with
        member this.GetEnumerator() = weights.GetEnumerator()

