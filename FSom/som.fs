namespace FSom

open System
open System.Linq
open System.Collections.Generic
open Microsoft.FSharp.Linq
open MathNet.Numerics
open MathNet.Numerics.Statistics
open MathNet.Numerics.LinearAlgebra.Generic
open MathNet.Numerics.LinearAlgebra.Double
open System.Threading
open System.Threading.Tasks
open System.Collections.Concurrent
open System.Linq

type Som(dims : int * int, nodes : Node seq) as this = 
    [<DefaultValue>]
    val mutable somMap : Node [,]
    let nodes = 
        match nodes with
        | null -> failwith "null array of nodes"
        | nodes -> nodes

    let dims = 
        match dims with
            | (x, y) when x = y && x > 0 -> (x, y)
            | _ -> failwith "wrong dimensions for the input som."
    let initialize () =
        let n = fst dims
        
        let nodeDim =
            match nodes.First().Count() with
            | 0 -> failwith "0-length node array"
            | n -> n

        let range = Array2D.zeroCreateBased 1 1 n n
        range |> Array2D.map (fun r -> Node(nodeDim))
    do
        this.somMap <- initialize()
    
    member  this.Dimensions = dims
    member this.Metric = Metric.Euclidian
    
    member this.GetBMU (node : Node) =
        let min = ref -1.
        let minI = ref -1
        let minJ = ref -1
        let distances = this.somMap |> Array2D.iteri (fun i j e -> 
            let dist = getDistance e node this.Metric
            if dist < !min then min := dist; minI := i; minJ := j)
        !minI, !minJ

    member this.GetBMUParallel (node : Node) =
        let monitor = new obj()
        let minList = ref []

        Parallel.ForEach(
            Partitioner.Create(0, fst dims), 
            (fun () -> (-1., -1, -1)), 
            (fun range state local -> 
                let mutable (min, minI, minJ) = 
                    match local with
                    | min, i, j -> min, i, j
                for i = fst range to snd range - 1 do
                    for j = 0 to snd this.Dimensions do
                        let dist = getDistance this.somMap.[i, j] node this.Metric
                        if dist < min then min <- dist; minI <- i; minJ <- j
                (min, minI, minJ)),
            (fun local -> lock monitor (fun () -> minList := local:: !minList))) |> ignore

        let minTuple = !minList |> List.minBy (fun (x, i, j) -> x)
        match minTuple with
        | x, i, j -> i, j
