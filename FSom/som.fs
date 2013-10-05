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
open System.IO

type Som(dims : int * int, nodes : Node seq) as this = 
    
    [<DefaultValue>] val mutable somMap : Node [,]
    let mutable metric = Metric.Euclidian

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

    let width, height = snd dims, fst dims

    let R0 = float((min width height)/ 2)
    let nrule0 = 0.9

    let modifyR x epochs =
        R0 * exp(-10.0 * (x * x) / float(epochs * epochs))
        
    let modifyTrainRule x epochs =
        nrule0 * exp(-10.0 * (x * x) / float(epochs * epochs))
    do
        this.somMap <- initialize()

    static member Read fileName =
        if not (File.Exists fileName) then failwith ("file does not exist: " + fileName)
        
        let lines = File.ReadAllLines fileName
        let name = ref String.Empty
        let classs = ref String.Empty
        let nodeLen = 0
        let nodes = List<Node>()

        lines 
        |> Seq.iteri 
            (fun i line -> 
                let nameAndClass = not (i &&& 0x1 = 1)
                let entries = line.Split([|' '; '\t'|])
                if nameAndClass then
                    name := entries.[0]
                    if entries.Length > 1 then
                        classs := entries.[1]
                else
                    let node = Node(entries |> Seq.map(fun e -> Double.Parse(e)))
                    if node.Count() <> nodeLen then failwith ("wrong length found, line " + i.ToString())
                    node.Name <- !name
                    node.Class <- !classs  
                    nodes.Add node
            )
        nodes.AsEnumerable()

    new (dim : int * int, fileName : string) = Som(dim, Som.Read fileName)
    
    member this.Width = width
    member this.Height = height
    member this.Metric 
        with get () =  metric
        and set value = metric <- value
    
    member this.InputNodes = nodes.ToList()
    member this.Item
        with get(i1, i2) = this.somMap.[i1, i2]

    member this.trainNode (trainingNode : Node) (originalNode : Node) (learningRule : float) =
        trainingNode |> Seq.iteri (fun i n -> trainingNode.[i] <- trainingNode.[i] + learningRule * (originalNode.[i] - trainingNode.[i]))

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
                    for j = 0 to width - 1 do
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

        let circCoords center R (boundary : int) = 
            let x1 = Math.Max(0, int(Math.Ceiling(float(center) - R)))
            let x2 = Math.Min(boundary, int(Math.Ceiling(float(center) + R)))
            x1, x2

        let circXY (ctr : int * int) R =
            let x1, x2 = circCoords (fst ctr) R (height - 1)
            let y1, y2 = circCoords (snd ctr) R (width - 1)
            x1, x2, y1, y2

        let totalEpochs = epochs
        let rec train R nrule iteration epochs isParallel =
            if epochs = 0 then ()
            else
                nodes |> Seq.iter 
                    (fun node ->
                        let (xBmu, yBmu) = this.GetBMUParallel(node)
                        let bmu = this.somMap.[xBmu, yBmu]
                    
                        if R <= 1.0 then                    
                            this.trainNode bmu node nrule
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
                                                this.trainNode this.somMap.[i,j] node (nrule * y)) |> ignore
                            else
                                for i = x1 to x2 do
                                    for j = y1 to y2 do
                                        let distSq = float((xBmu - i) * (xBmu - i) + (yBmu - j) * (yBmu - j))
                                        if distSq < RSq then
                                            let y = exp(-(10.0 * distSq) / (RSq))
                                            this.trainNode this.somMap.[i,j] node (nrule * y)
                    )
                let x = float(iteration + 1)
                train (modifyR x totalEpochs) (modifyTrainRule x totalEpochs) (iteration + 1) (epochs - 1) isParallel

        train R0 nrule0 0 epochs isParallel

    member this.InitClasses () =
        let classes = (this.somMap |> Seq.cast<Node> |> Seq.map (fun n -> n.Class)).Distinct()
        // randomly assign classes
        let rnd = Random(int32(DateTime.Now.Ticks))
        classes |> Seq.iter( fun c -> this.somMap.[rnd.Next(0, height), rnd.Next(0, width)].Class <- c)
        this.somMap |> Array2D.map(fun e -> e.Class)

    member this.Classify epochs =
        let totalEpochs = epochs
        this.InitClasses() |> ignore

        let rec classify epoch epochs rule = 
            if epochs = 0 then ()
            else
                nodes |> Seq.iter 
                    (fun node ->
                        let (xBmu, yBmu) = this.GetBMUParallel(node)
                        let bmu = this.somMap.[xBmu, yBmu]
                    
                        let y = if bmu.Class = node.Class then 1. else -1.                  
                        this.trainNode this.somMap.[xBmu, yBmu] node (rule * y)
                    )

                let x = float(epoch + 1)
                classify (epoch + 1) (epochs - 1) (modifyTrainRule x totalEpochs)

        classify 0 epochs nrule0
    