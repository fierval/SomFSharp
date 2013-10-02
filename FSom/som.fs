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
open Microsoft.FSharp.Collections

type Som(dims : int * int, nodes : Node seq) as this = 
    
    [<DefaultValue>] val mutable somMap : Node [,]
    let mutable metric = Metric.Euclidian
    let mutable epochs = 200

    let nodes = 
        match nodes with
        | null -> failwith "null array of nodes"
        | nodes -> nodes

    let dims = 
        match dims with
            | (x, y) when y > 0 && x > 0 -> (x, y)
            | _ -> failwith "wrong dimensions for the input som."
    let initialize () =
        let x,y = dims
        
        let nodeDim =
            match nodes.First().Count() with
            | 0 -> failwith "0-length node array"
            | n -> n

        let range = Array2D.zeroCreateBased 0 0 x y
        range |> Array2D.map (fun r -> Node(nodeDim))

    do
        this.somMap <- initialize()
    
    member  this.Dimensions = dims
    member this.Metric 
        with get () =  metric
        and set value = metric <- value

    member this.Item
        with get(i1, i2) = this.somMap.[i1, i2]

    member this.GetBMU (node : Node) =
        let min = ref Double.MaxValue
        let minI = ref -1
        let minJ = ref -1
        this.somMap |> Array2D.iteri (fun i j e -> 
            let dist = getDistance e node this.Metric
            if dist < !min then min := dist; minI := i; minJ := j)
        !minI, !minJ

    member this.GetBMUParallel (node : Node) =
        let monitor = new obj()
        let minList = ref []

        Parallel.ForEach(
            Partitioner.Create(0, fst dims), 
            (fun () -> ref (Double.MaxValue, -1, -1)), 
            (fun range state local -> 
                let mutable(min, minI, minJ) = 
                    match !local with
                    | m, i, j -> m, i, j
                for i = fst range to snd range - 1 do
                    for j = 0 to snd this.Dimensions - 1 do
                        let dist = getDistance this.somMap.[i, j] node this.Metric
                        if dist < min then 
                            min <- dist; minI <- i; minJ <- j
                local := (min, minI, minJ)
                local),
            (fun local -> lock monitor (fun () ->
                match !local with
                | m, i, j when i > 0 -> 
                    minList := (m, i, j) :: !minList
                |_ -> ()
                ))) |> ignore

        let minTuple = !minList |> List.minBy (fun (x, i, j) -> x)
        match minTuple with
        | x, i, j -> i, j
    
    member this.train(epochs, ?isParallel) =
        let isParallel = defaultArg isParallel true
        let R0 = float(fst this.Dimensions / 2)
        let nrule0 = 0.9

        let trainNode (trainingNode : Node) (originalNode : Node) (learningRule : float) =
            trainingNode |> Seq.iteri (fun i n -> trainingNode.[i] <- n + learningRule * (originalNode.[i] - n))

        let modifyR x =
            R0 * exp(-10.0 * (x * x) / float(epochs * epochs))
        
        let modifyTrainRule x =
            nrule0 * exp(-10.0 * (x * x) / float(epochs * epochs))

        let circCoords center R (boundary : int) = 
            let x1 = Math.Max(0, int(Math.Ceiling(float(center) - R)))
            let x2 = Math.Min(boundary, int(Math.Ceiling(float(center) + R)))
            x1, x2

        let circXY (ctr : int * int) R =
            let x1, x2 = circCoords (fst ctr) R (fst this.Dimensions - 1)
            let y1, y2 = circCoords (snd ctr) R (snd this.Dimensions - 1)
            x1, x2, y1, y2

        let rec train R nrule iteration epochs isParallel =
            if epochs = 0 then ()
            else
                nodes |> Seq.iter 
                    (fun node ->
                        let (xBmu, yBmu) = this.GetBMUParallel(node)
                        let bmu = this.somMap.[xBmu, yBmu]
                    
                        if R <= 1.0 then                    
                            trainNode bmu node nrule
                        else
                            let x1, x2, y1, y2 = circXY (xBmu, yBmu) R
                            let RSq = R * R
                            if isParallel then
                                Parallel.For(x1, x2 + 1, 
                                    fun i -> 
                                        for j = y1 to y2 do
                                            let distSq = float((xBmu - i) * (xBmu - i) + (yBmu - j) * (yBmu - j))
                                            if distSq < RSq then
                                                let y = exp(-(10.0 * distSq) / (RSq))
                                                trainNode this.somMap.[i,j] node (nrule * y)) |> ignore
                            else
                                for i = x1 to x2 do
                                    for j = y1 to y2 do
                                        let distSq = float((xBmu - i) * (xBmu - i) + (yBmu - j) * (yBmu - j))
                                        if distSq < RSq then
                                            let y = exp(-(10.0 * distSq) / (RSq))
                                            trainNode this.somMap.[i,j] node (nrule * y)
                    )
                let x = float(iteration + 1)
                train (modifyR x) (modifyTrainRule x) (iteration + 1) (epochs - 1) isParallel

        train R0 nrule0 0 epochs isParallel
            