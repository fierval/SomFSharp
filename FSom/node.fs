namespace FSom

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

    static let rndSource = new MersenneTwister()
    static let gamma = Gamma(2.0, 1.5)

    do
        match weights.Count with
        | 0 -> failwith "empty node"
        | _ -> ignore()

        this.classs <- ""
        this.name <- ""

    static do
        gamma.RandomSource <- rndSource


    new (len) = Node(gamma.Samples().Take(len))
    new (name, classs, (weights : float seq)) as this = 
        Node(weights)
        then
            this.name <- name
            this.classs <- classs
    
    member this.Display = weights.Aggregate("{", (fun a e -> a + e.ToString() + ", "), (fun a -> a.Substring(0, a.Length - 2) + "}"))
    member this.Class = this.classs
    member this.Name = this.name
            
    member this.Item 
        with get(index) = weights.[index]
        and set index value = weights.[index] <- value

    member this.Dimension = weights.Count

    /// Enumeration interfaces to enable
    /// treatment of node as a sequence of its values
    interface IEnumerable with
        member this.GetEnumerator() = weights.GetEnumerator() :> IEnumerator

    interface IEnumerable<float> with
        member this.GetEnumerator() = weights.GetEnumerator()

