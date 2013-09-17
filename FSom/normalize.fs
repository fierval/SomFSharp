namespace FSom

open System
open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra.Generic
open MathNet.Numerics.LinearAlgebra.Double
open Microsoft.FSharp.Collections
open System.Collections.Generic

open System.Linq

type Normalization =
    | NoNorm
    | MinMax
    | Zscore

[<AutoOpen>]
module normalize =
    // in order to get normalization constants, vectors are arranged in a matrix as its columns.
    // normalization constants are computed from the rows of the resulting matrix, as they 
    // now represent the actual series.
    let getNormalConstants (nodes : IList<float IList>) norm =
        let vectors  = nodes |> Seq.map(fun n -> DenseVector.ofSeq(n) :> Vector<float>) |> Seq.toArray
        let matrix =  DenseMatrix.OfColumnVectors(vectors)

        let rowEnum = matrix.RowEnumerator()
        let normalizable = rowEnum |> Seq.map (fun r -> snd r)

        match norm with
        | NoNorm -> [for i in [1..normalizable.Count()] -> (0., 1.)]
        | MinMax -> 
            let addMul = 
                normalizable 
                |> Seq.map (fun sq -> 
                    let min = -sq.Min()
                    min, // returns a tuple
                    let max = sq.Max() + min
                    if max = 0. then 0.8 else 0.8 / max) |> Seq.toList
            addMul
        | Zscore ->
            let addMul = 
                normalizable 
                |> Seq.map(fun l -> 
                    - l.Average(),  // returns a tuple
                    let std = l.StandardDeviation()
                    if std = 0. then 1. else 1. / std)  |> Seq.toList
            addMul

    let normalize (nodes : IList<float IList>) norm = 
        let addMul = getNormalConstants nodes norm
        for node in nodes do
            for i = 0 to node.Count - 1 do
                node.[i] <- (node.[i] + fst addMul.[i]) * snd addMul.[i]
        
        