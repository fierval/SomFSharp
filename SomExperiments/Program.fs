open FSom
open System.Linq
open System.Collections.Generic
open System
open System.Diagnostics
open System.IO

let stopWatch = Stopwatch()

let tic () = 
    stopWatch.Restart()

let toc () = 
        stopWatch.Stop()
        stopWatch.Elapsed.TotalMilliseconds

let mainTrainTest argv = 
    let bound = 12000
    let dim1 = 80
    let dim2 = 90
    let nodes = ([1..bound] |> Seq.map (fun i -> Node(11))).ToList()
    let som1 = SomGpu((dim1, dim2), nodes)

    printfn "training with massive iterations bmu\n"
    som1.Train 5 |> ignore

let main argv = 
    
    for j = 3 to 3 do
        let upper = 10. ** float(j)
        let bound = int(floor(upper * 1.2))
        let nodes = ([1..bound] |> Seq.map (fun i -> Node(12))).ToList()
        let som1 = SomGpu((200, 200), nodes)
        let map = (som1.somMap |> Seq.cast<Node>).SelectMany(fun (n : Node) -> n.AsEnumerable()).ToArray()
        printfn "\n"
        printfn "Number of nodes: %d" bound
        for i = 1 to 3 do
            printfn "\n"
            printfn "============================"
            printfn "\tAttempt: %d" i

            tic()
            let minsGpu2 = som1.GetBmuGpuUnified map nodes
            printfn "\tgpu iterations multiple copies of nodes: %10.3f ms" (toc())

            if j < 2 then
                tic()
                nodes |> Seq.iter( fun node -> som1.GetBMUParallel(node) |> ignore)
                printfn "\tcpu parallel: %10.3f ms" (toc())

                tic()
                nodes |> Seq.iter( fun node -> som1.GetBMU(node) |> ignore)
                printfn "\tcpu sequential: %10.3f ms" (toc())

                let rnd = Random(int(DateTime.Now.Ticks))
                let ind = rnd.Next(0, bound)
                tic()
                let min = som1.GetBMU(nodes.[ind])
                printfn "cpu timing for a single node: %10.3f ms" (toc())

let mainBmuTest argv = 
    let bound = 1200
    let dim1 = 30
    let dim2 = 20
    let nodes = ([1..bound] |> Seq.map (fun i -> Node(11))).ToList()
    let som1 = SomGpu((dim1, dim2), nodes)
    let map = (som1.somMap |> Seq.cast<Node>).SelectMany(fun (n : Node) -> n.AsEnumerable()).ToArray()

    let bmus = som1.GetBmuGpuUnified map nodes
    let mutable failed = 0
    for i = 0 to bound - 1 do
        let bmu = som1.GetBMU nodes.[i]
        let bmuSom = som1.toSomCoordinates bmus.[i]
        
        if bmu = bmuSom then 
            () //printfn "Success!"
        else
            printfn "ind: %d, bmu: %A, bmuSom: %A" i bmu bmuSom
            failed <- failed + 1
    printfn "failed: %d" failed

let classifyTrainTest argv = 
    let bound = 12000
    let dim1 = 80
    let dim2 = 90
    let classes = 21
    let nodes = ([1..bound] |> Seq.map (fun i -> Node(32))).ToList()
    let rnd = Random(int32(DateTime.Now.Ticks))
    nodes |> Seq.iter (fun n -> n.Class <- rnd.Next(0, classes).ToString())
    let som1 = SomGpu((dim1, dim2), nodes)
    for i = 1 to 3 do
        printfn "\n"
        printfn "============================"
        printfn "\tAttempt: %d" i

        tic()
        let minsGpu1 = som1.TrainClassifier 1
        printfn "\tgpu train classifier: %10.3f ms" (toc())

        tic()
        let minsGpu1 = som1.TrainClassifierLinear 1
        printfn "\tlinear train classifier: %10.3f ms" (toc())


let findDistance argv =
    let bound = 52000
    let dim1 = 200
    let dim2 = 200
    let nodes = ([1..bound] |> Seq.map (fun i -> Node(12))).ToList()
    let som1 = SomGpu((dim1, dim2), nodes)
    
    for i = 1 to 3 do
        printfn "\n"
        printfn "============================"
        printfn "\tAttempt: %d" i

        tic()
        let distGpu = som1.DistanceMap() 
        printfn "\tgpu distance map: %10.3f ms" (toc())

        tic()
        let dist = som1.LinearGetDistanceMap()
        printfn "\tlinear distance map: %10.3f ms" (toc())

        let rnd = Random(int(DateTime.Now.Ticks))
        let i = rnd.Next(0, dim1)
        let j = rnd.Next(0, dim2)

        printfn "\tlinear[%d, %d]=%10.5f, gpu[%d, %d]=%10.5f" i j dist.[i,j] i j distGpu.[i,j]

let shortMapTest argv =
    let bound = 120
    let dim1 = 5
    let dim2 = 5
    let nodes = ([1..bound] |> Seq.map (fun i -> Node(12))).ToList()
    let som1 = SomGpu((dim1, dim2), nodes)
    let nodes = ([1..bound] |> Seq.map (fun i -> Node(11))).ToList()
    let som1 = SomGpu((dim1, dim2), nodes)
    let map = (som1.somMap |> Seq.cast<Node>).SelectMany(fun (n : Node) -> n.AsEnumerable()).ToArray()

    let bmus = som1.GetBmuGpuUnified map nodes
    let mutable failed = 0
    for i = 0 to bound - 1 do
        let bmu = som1.GetBMU nodes.[i]
        let bmuSom = som1.toSomCoordinates bmus.[i]
        
        if bmu = bmuSom then 
            () //printfn "Success!"
        else
            printfn "ind: %d, bmu: %A, bmuSom: %A" i bmu bmuSom
            failed <- failed + 1
    printfn "failed: %d" failed

let timeShortMapTest argv =
    let bound = 12000
    let dim1 = 90
    let dim2 = 80
    let nodes = ([1..bound] |> Seq.map (fun i -> Node(12))).ToList()
    let som1 = SomGpu((dim1, dim2), nodes)
    let map = (som1.somMap |> Seq.cast<Node>).SelectMany(fun (n : Node) -> n.AsEnumerable()).ToArray()

    for i = 1 to 3 do
        printfn "\n"
        printfn "============================"
        printfn "\tAttempt: %d" i

        tic()
        let dist = som1.GetBmuGpuUnified map nodes
        printfn "\tgpu unified: %10.3f ms" (toc())

let bmuShortDistance (argv : string []) =
    let som1 = SomGpu((10, 30), argv.[0])
    som1.NormalizeInput(Normalization.Zscore)

    let map = som1.MergeNodes().ToArray()
    let nodes = som1.somMap |> Seq.cast<Node>

    let bmus = som1.GetBmuGpuUnified map nodes
    bmus

let pairwise (argv : string []) =
    let som = SomGpu((10, 30), argv.[0], 1)

    for i = 1 to 3 do
        tic()
        let dist1 = som.PairwiseDistance()
        printfn "Pairwise distance computed in: %10.3f ms" (toc())

let density (argv : string []) = 
    
    let buildStringSeq height (arr : string [,]) =
        seq {
            for i = 0 to height - 1 do 
                yield String (arr.[i..i, 0..] 
                    |> Seq.cast<string> |> Seq.fold (fun st e -> st + "\t" + e) String.Empty |> Seq.skip 1 |> Seq.toArray)
        }

    let som, nodes = SomGpu.ReadTrainSom argv.[0] argv.[1] 1
    let height, width = som |> Array2D.length1, som|> Array2D.length2
    let somGpu = SomGpu((height, width), nodes)
        
    somGpu.somMap <- som
    somGpu.NormalizeInput(Normalization.Zscore)

    for i = 1 to 1 do
        tic()
        let dense = somGpu.DensityMatrix()
        let strDensityMatrix = dense |> Array2D.map(fun e -> e.ToString()) |> buildStringSeq height

        let distOutput = List<string>()
        distOutput.AddRange strDensityMatrix
        File.WriteAllLines("proteomicsPBMC_mod_map_dense_map.txt", distOutput)

        printfn "Density matrix computed in: %10.3f ms" (toc())
        printfn "> 0: %d" (dense |> Seq.cast<int> |> Seq.filter (fun e -> e > 0) |> Seq.length)

let ustar (argv : string []) =
    let buildStringSeq height (arr : string [,]) =
        seq {
            for i = 0 to height - 1 do 
                yield String (arr.[i..i, 0..] 
                    |> Seq.cast<string> |> Seq.fold (fun st e -> st + "\t" + e) String.Empty |> Seq.skip 1 |> Seq.toArray)
        }

    let som, nodes = SomGpu.ReadTrainSom argv.[0] argv.[1] 1
    let height, width = som |> Array2D.length1, som|> Array2D.length2
    let somGpu = SomGpu((height, width), nodes)
    tic()    
    somGpu.somMap <- som
    somGpu.NormalizeInput(Normalization.Zscore)
    
    let dense = somGpu.DensityMatrix()
    let dist = somGpu.DistanceMap()
    let uMatrix = somGpu.UStarMatrix dist dense |> Array2D.map(fun e -> e.ToString()) |> buildStringSeq height
    let distOutput = List<string>()
    distOutput.AddRange uMatrix
    File.WriteAllLines("proteomicsPBMC_mod_map_ustar_map.txt", distOutput)

    printfn "U*-matrix computed in: %10.3f ms" (toc())



[<EntryPoint>]
let tests argv =
    //classifyTrainTest argv
    //mainTrainTest argv
    //for i = 0 to 10 do
    //mainBmuTest argv
    //main argv
    findDistance argv
    //shortMapTest argv
    //timeShortMapTest argv
    //bmuShortDistance argv |> ignore
    //pairwise argv
    //density argv
    //ustar argv
    0