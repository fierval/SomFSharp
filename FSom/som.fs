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
    [<DefaultValue>] val mutable asArray : float []

    let mutable metric = Metric.Euclidian
    let mutable shouldClassify = false

    let mutable nodes = 
        match nodes with
        | null -> failwith "null array of nodes"
        | nodes -> nodes.ToList()

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

    let getBlockDim len nThreads = (len + nThreads - 1) / nThreads

    let dimX, dimY = 32, 32

    let buildStringSeq (arr : string [,]) =
        seq {
            for i = 0 to this.Height - 1 do 
                yield String (arr.[i..i, 0..] 
                    |> Seq.cast<string> |> Seq.fold (fun st e -> st + "\t" + e) String.Empty |> Seq.skip 1 |> Seq.toArray)
        }

    let separate (output : System.Collections.Generic.List<string>) title = 
        if output.Count > 0 then
            output.Add(String.Empty)
        output.Add title
        output.Add("--------------------")
    
    let flattenSom (som : Node [,]) =
        let height, width = som |> Array2D.length1, som |> Array2D.length2
        let z = som.[0, 0].Count()
        let arr : float [] = Array.zeroCreate(width * height * z)
        som 
        |> Array2D.iteri (
            fun i j e -> 
            e 
            |> Seq.iteri (
                fun k el -> arr.[i * width * z + j * z + k] <- el))
        arr
    do
        this.somMap <- initialize()
        printfn "Initialized map: %d x %d, %d dimensions, %d nodes" height width (this.NodeLen) (nodes.Count)
        this.asArray <- flattenSom this.somMap
                    
    static member ParseNodes (lines : string []) =
        let seps = [|' '; '\t'|]
        let name = ref String.Empty
        let classs = ref String.Empty
        let nodeLen = ref 0
        let nodes = List<Node>()
        let format = 
            // determine formatting
            if lines.Length >= 2 then
                let fstLen = lines.[0].Split(seps).Length
                let sndLen = lines.[1].Split(seps).Length
                if fstLen <= 2 && sndLen > 2 then 0 //classes and nodes on separate lines
                else 1 // classes and nodes on the same line
            else
                1 // classes and nodes on the same line
                    
        lines 
        |> Seq.iteri 
            (fun i line -> 
                let entries = line.Split(seps)
                match format with
                | n when n = 0 ->
                    let nameAndClass = not (i &&& 0x1 = 1)
                    if nameAndClass then
                        name := entries.[0]
                        if entries.Length > 1 then
                            classs := entries.[1]
                    else
                        let node = Node(entries |> Seq.map(fun e -> Double.Parse(e)))
                        if node.Count() <> !nodeLen && !nodeLen > 0 then failwith ("wrong length found, line " + i.ToString())
                        if !nodeLen = 0 then
                            nodeLen := node.Count()
                        node.Name <- !name
                        node.Class <- !classs  
                        nodes.Add node
                | n when n = 1 ->
                    let res = ref 0.
                    let resInt = ref 0
                    let haveClasses = not (Double.TryParse(entries.[1], res)) && not (Int32.TryParse(entries.[1], resInt))
                    let haveName = not (Double.TryParse(entries.[0], res)) && not (Int32.TryParse(entries.[0], resInt))

                    let node = Node(entries |> Seq.skip(if haveClasses then 2 else (if haveName then 1 else 0)) |> Seq.map(fun e -> Double.Parse(e)))    
                    if node.Count() <> !nodeLen && !nodeLen > 0 then failwith ("wrong length found, line " + i.ToString())
                    if !nodeLen = 0 then
                        nodeLen := node.Count()
                    if haveName then
                        node.Name <- entries.[0]
                    if haveClasses then
                        node.Class <- entries.[1]
                    nodes.Add node
                | _ -> failwith "unsupported format"    
            )
        nodes

    static member ParseSom (lines : string []) nodeLen =
        Array2D.init (lines.Length) (lines.[0].Split('\t').Length / nodeLen) 
            (fun i j ->
                let entries = lines.[i].Split('\t')
                Node(Array.init nodeLen (fun k -> Double.Parse(entries.[j * nodeLen + k])))
            ) 

    static member ParseClasses (som : Node [,]) nodeLen (classLines : string []) =
        let width = som |> Array2D.length1        
        classLines 
        |> Seq.iteri
            (fun i c ->
                let x = i / width 
                let y = i - x * width
                som.[x, y].Class <- c
            )

    static member ReadTestSom somFileName nodesFileName header = 
        if not (File.Exists somFileName) then failwith ("file does not exist: " + somFileName)
        if not (File.Exists nodesFileName) then failwith ("file does not exist: " + nodesFileName)

        let seps = [|' '; '\t'|]
        let lines = File.ReadAllLines somFileName
        let nodesLines = (File.ReadAllLines nodesFileName).Skip(header).ToArray()
        
        let nodes = Som.ParseNodes nodesLines
        let nodeLen = nodes.[0].Count()

        let weightLines = lines.SkipWhile(fun l -> l <> "Classified Weights")
        if weightLines.Count() = 0 then failwith "Classifier not trained"
        let weightLines = weightLines.Skip(2).TakeWhile(fun l -> l <> String.Empty).ToArray()
        let som = Som.ParseSom weightLines nodeLen
        
        let classLines = lines.SkipWhile(fun l -> l <> "Classes").Skip(2).TakeWhile(fun l -> l <> String.Empty).ToArray()
        Som.ParseClasses som nodeLen classLines

        som, nodes

    static member ReadTrainSom somFileName nodesFileName header =
        if not (File.Exists somFileName) then failwith ("file does not exist: " + somFileName)
        if not (File.Exists nodesFileName) then failwith ("file does not exist: " + nodesFileName)

        let seps = [|' '; '\t'|]
        let lines = File.ReadAllLines somFileName
        let nodesLines = (File.ReadAllLines nodesFileName).Skip(header).ToArray()
        
        let nodes = Som.ParseNodes nodesLines
        let nodeLen = nodes.[0].Count()

        let weightLines = lines.SkipWhile(fun l -> l <> "Trained Weights")
        if weightLines.Count() = 0 then failwith "Did not find trained weights"
        let weightLines = weightLines.Skip(2).TakeWhile(fun l -> l <> String.Empty).ToArray()
        let som = Som.ParseSom weightLines nodeLen
        
        som, nodes
        
    static member Read fileName header =
        if not (File.Exists fileName) then failwith ("file does not exist: " + fileName)


        let lines = if header > 0 then (File.ReadAllLines fileName).Skip(header).ToArray() else File.ReadAllLines fileName
        Som.ParseNodes lines


    new (dim : int * int, fileName : string, ?header) = 
        let header = defaultArg header 0
        Som(dim, Som.Read fileName header) 
        

    member this.ModifyTrainRule x epochs =
        nrule0 * exp(-10.0 * (x * x) / float(epochs * epochs))
        
    member this.toSomCoordinates i =
        let x = i / this.Width 
        let y = i - x * this.Width
        x, y

    member this.ShouldClassify 
        with get() = this.InputNodes.Where(fun (n : Node) -> not (String.IsNullOrEmpty(n.Class))).Count() > 0
    
    member this.toArray = flattenSom this.somMap

    member this.GetBlockDim len nThreads = getBlockDim len nThreads
    member this.DimX = dimX
    member this.DimY = dimY

    member this.NodeLen = this.somMap.[0,0].Count()
    member this.Width = width
    member this.Height = height
    member this.Metric 
        with get () =  metric
        and set value = metric <- value
    
    member this.NormalizeInput (norm : Normalization) =
        nodes <- 
            (normalize(this.InputNodes.Select(fun (n : Node) -> n.ToList() :> IList<float>).ToList())  norm
            |> Seq.map2 (
                fun (old : Node) (n : IList<float>) 
                    -> 
                        let node = Node(n)
                        node.Class <- old.Class
                        node.Name <- old.Name
                        node) nodes).ToList()
            

    member this.InputNodes : System.Collections.Generic.List<Node> = nodes
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
            Partitioner.Create(0, this.Height), 
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
                | m, i, j -> 
                    minList := (m, i, j) :: !minList
                ))) |> ignore

        let minTuple = !minList |> List.minBy (fun (x, i, j) -> x)
        match minTuple with
        | x, i, j -> i, j
    
    abstract member Train : int -> Node [,]
    default this.Train epochs =
        this.train epochs

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
            if epochs = 0 then this.somMap
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
                                                let y = exp(-(1.0 * distSq) / (RSq))
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

    abstract member InitClasses : unit -> unit
    default this.InitClasses () =
        let classes = (nodes |> Seq.map (fun n -> n.Class)).Distinct().Where(fun c -> not (String.IsNullOrEmpty(c))).ToList()
        //this.somMap |> Seq.cast<Node> |> Seq.iter2 (fun c e -> e.Class <- c) classes
        this.somMap |> Array2D.iteri(fun i j e -> e.Class <- classes.[(i * this.Width + j) % classes.Count])

    abstract member TrainClassifier : int -> unit
    default this.TrainClassifier epochs = this.TrainClassifierLinear epochs

    member this.TrainClassifierLinear epochs =
        let totalEpochs = epochs
        this.InitClasses()

        let rec classify epoch epochs rule = 
            if epochs = 0 then ()
            else
                for node in nodes do 
                    let (xBmu, yBmu) = this.GetBMUParallel(node)
                    let bmu = this.somMap.[xBmu, yBmu]
                    
                    if not (String.IsNullOrEmpty(bmu.Class)) then
                        let y = if bmu.Class = node.Class then 1. else -1.               
                        this.trainNode this.somMap.[xBmu, yBmu] node (rule * y)

                let x = float(epoch + 1)
                classify (epoch + 1) (epochs - 1) (modifyTrainRule x totalEpochs)

        classify 0 epochs nrule0
    
    member this.LinearGetDistanceMap () =
        let distMap = Array2D.zeroCreate (this.somMap |> Array2D.length1) (this.somMap |> Array2D.length2)
        this.somMap 
        |> Array2D.iteri 
            (fun i j e ->
                let dist = ref 0.
                let n = ref 0.
                for x1 = i - 1 to i + 1 do
                    for y1 = j - 1 to j + 1 do
                        if x1 >= 0 && y1 >= 0 && x1 < this.Height && y1 < this.Width && (x1 <> i || y1 <> j) then
                            n := !n + 1.
                            let thisDist = ref 0.
                            this.somMap.[i, j] |> Seq.iter2 (fun n1 n2 -> thisDist := !thisDist + (n1 - n2) ** 2.) this.somMap.[x1, y1]
                            dist := !dist + sqrt !thisDist
                distMap.[i, j] <- !dist / !n
            )
        distMap
    
    abstract member DistanceMap : unit -> float [,]
    default this.DistanceMap () =
        this.LinearGetDistanceMap()

    abstract member Classify : Node seq -> string []
    default this.Classify nodes =
        nodes 
        |> Seq.map (fun n ->
            let x, y = this.GetBMUParallel(n)
            this.somMap.[x,y].Class
            )
        |> Seq.toArray

    // save to file. if distClassSeparate = true
    // saves distance map and class map to separate files
    member this.Save epochs fileName distClassSeparate =
        if String.IsNullOrWhiteSpace fileName then failwith "File name must be specified"

        if File.Exists fileName then File.Delete fileName

        let output = List<string>()

        let saveWeights (arr : Node [,]) = 
            Array2D.init this.Height (this.Width * this.NodeLen )
                (fun i j ->
                    arr.[i, j / this.NodeLen].[ j % this.NodeLen].ToString()
                ) |> buildStringSeq
                 
        let insertIntoFileName fileName insert =
            let path, name, ext = Path.GetDirectoryName(fileName), Path.GetFileNameWithoutExtension(fileName) + insert, Path.GetExtension(fileName)
            Path.Combine(path, name + ext)

        // 2D weights. Weights are calc'ed first before classifier has mucked them up
        let weights = saveWeights this.somMap

        let distMap = this.DistanceMap()
        let strDistMap = distMap |> Array2D.map(fun e -> e.ToString()) |> buildStringSeq

        // distance map (U-matrix)
        if distClassSeparate then 
            let distOutput = List<string>()
            distOutput.AddRange strDistMap
            File.WriteAllLines(insertIntoFileName fileName "_dist_map", distOutput)
        else
            separate output "Distance Map"
            output.AddRange(strDistMap)

        // density matrix (P-matrix)
        let denseMap = this.DensityMatrix()
        let strDensityMatrix = denseMap |> Array2D.map(fun e -> e.ToString()) |> buildStringSeq

        if distClassSeparate then 
            let distOutput = List<string>()
            distOutput.AddRange strDensityMatrix
            File.WriteAllLines(insertIntoFileName fileName "_dense_map", distOutput)
        else
            separate output "Density Matrix"
            output.AddRange(strDensityMatrix)

        // U*-matrix
        let uStarMatrix = this.UStarMatrix distMap denseMap
        let strUStarMatrix = uStarMatrix |> Array2D.map(fun e -> e.ToString()) |> buildStringSeq

        if distClassSeparate then 
            let distOutput = List<string>()
            distOutput.AddRange strUStarMatrix
            File.WriteAllLines(insertIntoFileName fileName "_ustar_map", distOutput)
        else
            separate output "U* Matrix"
            output.AddRange(strDensityMatrix)

        // classification
        if this.ShouldClassify then 
            separate output "Classes"

            this.TrainClassifier epochs
            let classes = this.somMap |> Array2D.map (fun node -> node.Class) |> buildStringSeq
        
            output.AddRange classes

            if distClassSeparate then 
                let distOutput = List<string>()
                distOutput.AddRange classes
                File.WriteAllLines(insertIntoFileName fileName "_class_map", distOutput)

        separate output "Trained Weights"

        output.AddRange weights
        
        if this.ShouldClassify then
            separate output "Classified Weights"
            output.AddRange (saveWeights this.somMap)

        // write it all out
        File.WriteAllLines(fileName, output)

    member this.SaveClassified (nodes : Node seq) (classes : string []) outFile =
        if File.Exists outFile then File.Delete outFile
        if nodes = null || nodes.Count() = 0 || classes = null || classes.Length = 0 then failwith "Must supply non-empty classes and nodes arrays"
        
        let total = nodes.Count()
        let output = List<string>()
        let nodesClasses = nodes |> Seq.map2(fun cl n -> cl, n) classes
        let accurate = 
            nodesClasses 
            |> Seq.filter(
                fun (actual, expected) -> 
                    if not(String.IsNullOrWhiteSpace(expected.Class)) then String.Compare(actual, expected.Class) = 0 else true)
            |> Seq.length

        let accuracy = String.Format("{0:0.###}", float(accurate) / float(total))

        output.AddRange(
            nodesClasses 
            |> Seq.map (
                fun (cl, n) -> 
                    String(n |> Seq.fold(fun acc e -> acc + "\t" + e.ToString()) String.Empty |> Seq.skip 1 |> Seq.toArray) 
                    + "\t" + (if cl = n.Class then "+" + "\t" + cl else cl))) 
                    
        separate output "Accuracy"
        output.Add(accuracy)

        File.WriteAllLines(outFile, output)

    abstract PairwiseDistance : unit -> float []
    default this.PairwiseDistance () =
        let nodes = if this.InputNodes.Count > 1000 then this.InputNodes.Take(1000).ToList() else this.InputNodes
        let len = nodes.Count
        let distMatrix = Array.zeroCreate (len * (len - 1) / 2)

        Parallel.ForEach(nodes, 
            (fun node st (i : int64) -> 
                let i = int i
                for j =  i + 1 to nodes.Count - 1 do
                    distMatrix.[i * nodes.Count + j - (i + 1) * (i + 2) / 2] <- getDistanceEuclidian node this.InputNodes.[j]
            )
        ) |> ignore

        distMatrix

    member this.ParetoRadius =
        let distMatrix = this.PairwiseDistance()
        let percentile = Percentile(distMatrix)
        let paretoPercentile = percentile.Compute(0.2013)
        paretoPercentile //* 0.127417

    
    abstract DensityMatrix : unit -> int [,]
    default this.DensityMatrix () =
        let radius = this.ParetoRadius
        let densityMatrix = Array2D.zeroCreate this.Height this.Width

        Parallel.For(0, this.Height, 
            (
                fun i st ->
                    for j = 0 to this.Width - 1 do
                        let neuron = this.somMap.[i, j]
                        this.InputNodes 
                        |> Seq.iter (fun node ->
                            if getDistanceEuclidian neuron node < radius then
                                densityMatrix.[i, j] <- densityMatrix.[i, j] + 1)
            )
        ) |> ignore

        densityMatrix

    member this.ComputeUMatrixScalingFactors distanceMatrix densityMatrix =
        let seqP = densityMatrix |> Seq.cast<int> |> Seq.map (fun e -> float e)
        let vec =  DenseVector.ofSeq(seqP)
        let minP = vec.Min()
        let avgP =vec.Average()
        let divisor = avgP - minP
        divisor, minP

    abstract UStarMatrix : float [,] -> int [,] -> float [,]
    default this.UStarMatrix distanceMatrix densityMatrix = 
        let divisor, min = this.ComputeUMatrixScalingFactors distanceMatrix densityMatrix

        let res =
            distanceMatrix |> Array2D.mapi(fun i j d -> d * (float densityMatrix.[i, j] - min) / divisor)
        res
        