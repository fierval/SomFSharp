namespace FSom

open System
open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra.Generic
open MathNet.Numerics.LinearAlgebra.Double

open System.Linq

type Normalization =
    | MinMax
    | Zscore

[<AutoOpen>]
module normalize =

    let getNormalConstants (nodes : seq<float list>) norm =
        let vectors  = nodes |> Seq.map(fun n -> DenseVector.ofSeq(n) :> Vector<float>) |> Seq.toArray
        let matrix =  DenseMatrix.OfColumnVectors(vectors)

        let rowEnum = matrix.RowEnumerator()
        let normalizable = rowEnum |> Seq.map(fun r -> (snd r).Where(fun m -> m <> 0.).ToList())

        match norm with
        | MinMax -> 
            let add = normalizable.Min().Select(fun i -> -i) |> Seq.toList
            let max = normalizable.Max()
            let mul = [1..add.Length] |> List.map(fun i -> 0.9 / (max.[i] - add.[i]))
            mul, add
        | Zscore ->
            let add = normalizable |> Seq.map(fun l -> - l.Average())  |> Seq.toList
            let mul = normalizable |> Seq.map( fun l -> l.StandardDeviation()) |> Seq.toList
            mul, add

    