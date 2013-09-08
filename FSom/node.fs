namespace FSom

open System
open System.Collections
open System.Collections.Generic
open MathNet.Numerics
open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra.Generic
open MathNet.Numerics.LinearAlgebra.Double

type Node(weights : float seq) =
    let weights = DenseVector.OfEnumerable(weights)

    do
        match weights.Count with
        | 0 -> failwith "empty node"
        | _ -> ignore()

    static member private rnd = Random(int(DateTime.Now.Ticks))

    new (len) = Node([1..len] |> List.map (fun i -> Node.rnd.NextDouble()))
        
    member this.Item 
        with get(index) = weights.[index]

    member this.Dimension = weights.Count

    /// Enumeration interfaces to enable
    /// treatment of node as a sequence of its values
    interface IEnumerable with
        member this.GetEnumerator() = weights.GetEnumerator() :> IEnumerator

    interface IEnumerable<float> with
        member this.GetEnumerator() = weights.GetEnumerator()

