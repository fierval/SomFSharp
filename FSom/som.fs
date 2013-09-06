namespace FSom

open System
open System.Linq
open Microsoft.FSharp.Linq

type Som(dims : int * int, nodes : Node seq) = 
    let nodes = 
        match nodes with
        | null -> failwith "null array of nodes"
        | nodes -> nodes

    let dims = 
        match dims with
            | (x, y) when x = y && x > 0 -> (x, y)
            | _ -> failwith "wrong dimensions for the input som."

    let mutable somMap = Array2D.zeroCreate 1 1
    let initialize () =
        let n = fst dims
        
        let nodeDim =
            match nodes |> Seq.length with
            | 0 -> failwith "0-length node array"
            | n -> n

        let range = Array2D.zeroCreateBased 1 1 n n
        range |> Array2D.map (fun r -> Node(nodeDim))
    do
        somMap <- initialize()
    
    member  this.Dimensions = dims
    
