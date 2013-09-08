namespace FSom

open System
open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra.Generic
open MathNet.Numerics.LinearAlgebra.Double
open Microsoft.FSharp.Collections
open System.Collections.Generic

open System.Linq

type Metric =
    | Euclidian
    | Taxicab

[<AutoOpen>]
module distance =

    let getDistanceEuclidian (x : Node) (y : Node) =
        Math.Sqrt([0..x.Dimension - 1] |> Seq.fold(fun sq i -> sq + (x.[i] - y.[i]) ** 2.) 0.)

    let getDistanceTaxicab (x : Node) (y : Node) =
        let dim = x.Dimension
        [0..x.Dimension - 1] |> Seq.fold(fun sq i -> sq + Math.Abs(x.[i] - y.[i])) 0.

    let getDistance (x : Node) (y : Node) metric =
        if x.Dimension <> y.Dimension then failwith "Dimensions must match"
        else
            match metric with
            | Euclidian ->
                getDistanceEuclidian x y
            | Taxicab -> 
                getDistanceTaxicab x y