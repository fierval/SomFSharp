namespace FSom

open System
open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra.Generic
open MathNet.Numerics.LinearAlgebra.Double
open Microsoft.FSharp.Collections
open System.Collections.Generic

open System.Linq

type Normalization =
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
        let normalizable = (rowEnum |> PSeq.map (fun r -> snd r) |> PSeq.map(fun r -> r.Where(fun m -> m <> 0.)))

        match norm with
        | MinMax -> 
            let addMul = 
                normalizable 
                |> PSeq.map (fun sq -> 
                    let min = -sq.Min()
                    min, // returns a tuple
                    let max = sq.Max() + min
                    if max = 0. then 1. else 1. / max) |> Seq.toList
            addMul
        | Zscore ->
            let addMul = 
                normalizable 
                |> PSeq.map(fun l -> 
                    - l.Average(),  // returns a tuple
                    let std = l.StandardDeviation()
                    if std = 0. then 1. else std)  |> Seq.toList
            addMul

    let normalize (nodes : IList<float IList>) norm = 
        let addMul = getNormalConstants nodes norm
        nodes |> PSeq.map (fun n -> n |> Seq.map(fun e i -> (e + fst addMul.[i]) * snd addMul.[i]))
        