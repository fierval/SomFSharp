namespace FSom

open System
open MathNet.Numerics
open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra.Generic
open MathNet.Numerics.LinearAlgebra.Double

[<AutoOpen>]
module SomNode =
    let rnd = Random(int(DateTime.Now.Ticks))

    type Node(weights : float list) =
        let weights = DenseVector.OfEnumerable(weights)
        do
            match weights.Count with
            | 0 -> failwith "empty node"
            | _ -> ignore()

        new (len) = Node([1..len] |> List.map (fun i -> rnd.NextDouble()))
            
