﻿namespace FSom
open Alea.CUDA
open Alea.CUDA.Utilities
open System
open System.Diagnostics
open System.Linq
open System.Collections.Generic
open Microsoft.FSharp.Quotations

type SomGpuBase(dims, nodes : Node seq) =
    inherit Som(dims, nodes)

    let stopWatch = Stopwatch()

    let tic () = 
        stopWatch.Restart()

    let toc () = 
            stopWatch.Stop()
            stopWatch.Elapsed.TotalMilliseconds

    new (dim : int * int, fileName : string, ?header) =
        let header = defaultArg header 0
        SomGpuBase(dim, Som.Read fileName header)

    member this.pTrain = 
        cuda {
            let! kernelTrain = 
                <@ fun nodeLen len width height bmuX bmuY rSq nRule
                    (node : deviceptr<float>) 
                    (map :  deviceptr<float>) 
                    ->

                    let i = blockDim.x * blockIdx.x + threadIdx.x
                    if i < len then    
                        let x = i / width / nodeLen
                        let y = (i - x * width * nodeLen) / nodeLen
                        let z = i % nodeLen

                        let distSq = float((bmuX - x) * (bmuX - x) + (bmuY - y) * (bmuY - y))
                        if distSq < rSq then
                            map.[i] <- map.[i] + nRule * exp(-(1.0 * distSq) / (rSq)) * (node.[z] - map.[i])                                    

                @> |> Compiler.DefineKernel

            return Entry(fun (m:Program)  ->   
                let kernelTrain = m.Apply kernelTrain

                let run (epoch : int) epochs r nrule (nt : int) nBlocks (mins : int []) nodeLen len (dNodes : deviceptr<float>) (dMap : deviceptr<float>) = 
                    let lp = LaunchParam(nBlocks, nt)
                    for i = 0 to mins.Length - 1 do
                        let bmu = mins.[i]
                        let bmuX, bmuY = this.toSomCoordinates bmu
                        if r > 1. then
                            kernelTrain.Launch lp nodeLen len this.Width this.Height bmuX bmuY (r * r) nrule (dNodes + i * nodeLen) dMap
                        else 
                            this.trainNode this.somMap.[bmuX, bmuY] this.InputNodes.[i] nrule
                run
            )            
        }

    member this.pDistanceMap =
        cuda {
            let! kernelDistanceMap = 
                <@ fun nodeLen width height 
                    (map : deviceptr<float>)
                    (distMap : deviceptr<float>) ->
                        
                    let i = blockDim.x * blockIdx.x + threadIdx.x 

                    if i < width * height then 
                        let x = i / width
                        let y = i - x * width

                        let mutable dist = 0.
                        let mutable n = 0

                        for x1 = x - 1 to x + 1 do
                            for y1 = y - 1 to y + 1 do
                                if x1 >= 0 && y1 >= 0 && x1 < height && y1 < width && (x1 <> x || y1 <> y) then
                                    let j = x1 * width * nodeLen + y1 * nodeLen
                                    n <- n + 1
                                    let mutable thisDist = 0.
                                    for z = 0 to nodeLen - 1 do
                                        thisDist <- thisDist + (map.[i * nodeLen + z] - map.[j + z]) * (map.[i * nodeLen + z] - map.[j + z])
                                    dist <- dist + sqrt thisDist

                        distMap.[i] <- dist / float(n)

                @> |> Compiler.DefineKernel
        return 

            Entry (fun (m : Program) ->
                let pDistMap = m.Apply kernelDistanceMap
                let run () =  
                    
                    let len = this.asArray.Length
                    let nodeLen = this.NodeLen
                    let mapLen = len / nodeLen
                    let nt =  min (this.DimX * this.DimY) mapLen
                    let nBlocks = this.GetBlockDim mapLen nt

                    use dMap = m.Worker.Malloc(this.toArray)
                    use dDists = m.Worker.Malloc<float>(mapLen)
                    
                    let lp = LaunchParam(nBlocks, nt) 

                    pDistMap.Launch lp nodeLen this.Width this.Height dMap.Ptr dDists.Ptr
                    dDists.Gather()
                run
            )

        }
    // training
    member this.pDist = 
        cuda {
            let! kernel =
                <@ fun len nodeLen totalNodes iter
                    
                    (nodes : deviceptr<float>) 
                    (map :  deviceptr<float>)
                    (distances : deviceptr<float>)
                    (minDist : deviceptr<float>) 
                    (minIndex : deviceptr<int>)
                    ->
                    
                    let fit = len / nodeLen / totalNodes // how many times the nodes array "fits" the map array
                    let remainderCutoff = (len / nodeLen) %  totalNodes // iteration which will get redundant values for the part of the nodes array that doesn't "fit"

                    let shift = iter * nodeLen
                    let start = blockIdx.x * blockDim.x

                    let mutable cell = start + threadIdx.x

                    if cell < len then
                        let mutable cell = start + threadIdx.x
                        
                        let foldedNodeCell = cell / nodeLen // node "mapped" to the length of the map
                        let nodeCell = cell / nodeLen % totalNodes // actual node
                        
                        if cell < len then
                            cell <- cell + shift
                            if cell >= len then
                                cell <- cell - len

                            let nodeShift = cell % nodeLen
                            if foldedNodeCell < totalNodes * fit || iter < remainderCutoff then
                                distances.[cell] <- 
                                    (map.[cell] - nodes.[nodeCell * nodeLen + nodeShift]) 
                                    * (map.[cell] - nodes.[nodeCell * nodeLen + nodeShift])

                                // first threads in the block wrap it up for everyone
                                if nodeShift = 0 then
                                    __syncthreads()

                                    let mapCell = cell / nodeLen
                                    let mutable distSq = distances.[cell]
                                    for j = 1 to nodeLen - 1 do
                                        distSq <- distSq + distances.[cell + j]
                                    if minDist.[foldedNodeCell] > distSq || iter = 0 then
                                        minDist.[foldedNodeCell] <- distSq
                                        minIndex.[foldedNodeCell] <- mapCell
                @> |> Compiler.DefineKernel

            return Entry(
                fun 
                    (m:Program) 
                    ->
                let kernel = m.Apply kernel

                let run (dMap : DeviceMemory<float>) 
                    (dNodes : DeviceMemory<float>) 
                    (dTemp : DeviceMemory<float>)
                    (dMinDists : DeviceMemory<float>) 
                    (dIndex : DeviceMemory<int>)
                    len
                    nodeLen 
                    nNodes 
                    (nt : int) 
                    (nBlocks : int) =

                    let mapLen = len / nodeLen

                    let lp = LaunchParam(nBlocks, nt) //|> Engine.setDiagnoser diagnose

                    let fit = len / nodeLen / nNodes                    
                    for iter = 0 to len / nodeLen / fit - 1 do
                        kernel.Launch lp len nodeLen nNodes iter dNodes.Ptr dMap.Ptr dTemp.Ptr dMinDists.Ptr dIndex.Ptr

                    let finalMinIndex = Array.zeroCreate nNodes
                
                    let minDist = dMinDists.Gather()
                    let minIndex = dIndex.Gather()

                    for i in [0..nNodes - 1] do
                        let stride = nNodes
                        let mutable min = minDist.[i]
                        finalMinIndex.[i] <- minIndex.[i]
                        let mutable j = i + stride

                        while j < mapLen do
                            if min > minDist.[j] then
                                min <- minDist.[j]
                                finalMinIndex.[i] <- minIndex.[j]
                            j <- j + stride
                    finalMinIndex        
                run
            )
        }
    
    member this.pDistShortMap =
        cuda {
            let! kernel = 
                <@  fun mapLen nodeLen totalNodes iter
                    
                        (nodes : deviceptr<float>) 
                        (map :  deviceptr<float>)
                        (distances : deviceptr<float>)
                        (minDist : deviceptr<float>) 
                        (minIndex : deviceptr<int>) ->

                    let xNode = blockIdx.x * blockDim.x + threadIdx.x

                    if xNode < totalNodes * nodeLen then
                        let xMap = (xNode / nodeLen % mapLen + iter) % mapLen * nodeLen + threadIdx.x % nodeLen
                        distances.[xNode] <- (nodes.[xNode] - map.[xMap]) * (nodes.[xNode] - map.[xMap])

                        if threadIdx.x % nodeLen = 0 then
                            __syncthreads()
                            let mutable dist = 0.
                            for j = 0 to nodeLen - 1 do
                                dist <- dist + distances.[xNode + j]
                            if dist < minDist.[xNode / nodeLen] then
                                minDist.[xNode / nodeLen] <- dist
                                minIndex.[xNode / nodeLen] <- xMap / nodeLen

                @> |> Compiler.DefineKernel

            return Entry(
                fun 
                    (m:Program) 
                    ->
                let kernel = m.Apply kernel
                let run
                    (dMap : DeviceMemory<float>) 
                    (dNodes : DeviceMemory<float>) 
                    (dTemp : DeviceMemory<float>)
                    (dMinDists : DeviceMemory<float>) 
                    (dIndex : DeviceMemory<int>) 
                    len
                    nodeLen 
                    nNodes 
                    (nt : int) 
                    (nBlocks : int) 
                    =
                        let mapLen = len / nodeLen

                        let lp = LaunchParam(nBlocks, nt) 
                    
                        for iter = 0 to mapLen - 1 do
                            kernel.Launch lp mapLen nodeLen nNodes iter dNodes.Ptr dMap.Ptr dTemp.Ptr dMinDists.Ptr dIndex.Ptr
                
                        dIndex.Gather()
                run
                )
        }

    member this.pTrainSom = 
        cuda {
               
            return Entry(fun (m: Program) ->
                
                let run (nodes : float [] list) epochs =
                    use pdist = this.pDist  |> Compiler.load Worker.Default 
                    use pdistShortMap = this.pDistShortMap  |> Compiler.load Worker.Default 
                    use ptrain = this.pTrain  |> Compiler.load Worker.Default 

                    let nNodes = nodes.Length
                    let nodeLen = nodes.[0].Length
                    let mapLen = this.asArray.Length / nodeLen

                    let len = if nNodes <= mapLen then this.asArray.Length else nNodes * nodeLen

                    let nt =  ((this.DimX * this.DimY) / nodeLen) * nodeLen
                    let nBlocks = this.GetBlockDim len nt //split the array of nodes into blocks

                    use dMap = m.Worker.Malloc(this.toArray)
                    let minDist = Array.create nNodes Double.MaxValue
                    use dMinDists = if nNodes <= mapLen then m.Worker.Malloc<float>(mapLen) else m.Worker.Malloc(minDist)
                    use dMinIndex = m.Worker.Malloc<int>(if nNodes <= mapLen then mapLen else nNodes) 
                    use dTemp = m.Worker.Malloc<float>(len)
                    use dNodes = m.Worker.Malloc(nodes.SelectMany(fun n -> n :> float seq).ToArray())

                    let program =
                        (if nNodes <= mapLen then pdist else pdistShortMap) 
                    // training constants                    
                    let R0 = float((max this.Height this.Width) / 2)
                    let nrule0 = 0.9
                    let modifyR x =
                        R0 * exp(-10.0 * (x * x) / float(epochs * epochs))
        
                    let modifyTrainRule x =
                            nrule0 * exp(-10.0 * (x * x) / float(epochs * epochs))

                    // training is in single dimension because alea.cuBase doesn't
                    // handle multiple dimensions.
                    let mapFullLen = this.asArray.Length
                    let nBlocksTrain = this.GetBlockDim mapFullLen nt
                    m.Worker.ProfilerStart()
                    let getMins () =
                        program.Run dMap dNodes dTemp dMinDists dMinIndex mapFullLen nodeLen nNodes nt nBlocks

                    for epoch = 0 to epochs - 1 do 
                        // Step 1. Find the BMUs
                        tic ()

                        let mins = getMins()
                        //printfn "found all minimums for the epoch: %10.3f ms" (toc())
                        //tic()

                        let r = modifyR (float epoch)
                        let nrule = modifyTrainRule (float epoch)

                        //Step 2. Training.
                        ptrain.Run epoch epochs r nrule nt nBlocksTrain mins nodeLen mapFullLen dNodes.Ptr dMap.Ptr
                        printfn "epoch: %d, nrule: %10.5f, R: %10.3f, time: %10.3f ms" epoch nrule r (toc())

                    let res = dMap.Gather()
                    m.Worker.ProfilerStop()
                    res
                run
            )    
        }

        member this.distanceModules =
            let options = {CompileOptions.Release with ModuleName = "dist"}
            let irmDist = Compiler.Compile(this.pDist, options).IRModule
            let options = {CompileOptions.Release with ModuleName = "distShort"}
            let irmDistShort = Compiler.Compile(this.pDistShortMap, options).IRModule
            
            irmDist, irmDistShort

        member this.pTestUnifiedBmu = 
            cuda {
                let pdist = this.pDist
                let pdistShortMap = this.pDistShortMap

                return Entry(
                        fun (m : Program) ->
                        
                    let run (map : float []) (nodes : float [] list) =

                        use pdist = pdist |> Compiler.load Worker.Default
                        use pdistShortMap = pdistShortMap  |> Compiler.load Worker.Default

                        let nNodes = nodes.Length
                        let nodeLen = nodes.[0].Length
                        let mapLen = map.Length / nodeLen

                        let len = if nNodes <= mapLen then map.Length else nNodes * nodeLen

                        let nt =  ((this.DimX * this.DimY) / nodeLen) * nodeLen
                        let nBlocks = this.GetBlockDim len nt //split the array of nodes into blocks

                        use dMap = m.Worker.Malloc(map)
                        let minDist = Array.create nNodes Double.MaxValue
                        use dMinDists = if nNodes <= mapLen then m.Worker.Malloc<float>(mapLen) else m.Worker.Malloc(minDist)
                        use dMinIndex = m.Worker.Malloc<int>(if nNodes <= mapLen then mapLen else nNodes) 
                        use dTemp = m.Worker.Malloc<float>(len)
                        use dNodes = m.Worker.Malloc(nodes.SelectMany(fun n -> n :> float seq).ToArray())

                        if nNodes <= mapLen  then
                            pdist.Run dMap dNodes dTemp dMinDists dMinIndex map.Length nodeLen nNodes nt nBlocks
                        else
                            pdistShortMap.Run dMap dNodes dTemp dMinDists dMinIndex map.Length nodeLen nNodes nt nBlocks
                    run
                    )
            }

        member this.pTrainClassifier =
            cuda {
                let pdist = this.pDist
                let pdistShortMap = this.pDistShortMap

                return Entry(
                        fun (m : Program) ->
                    use pdist = pdist |> Compiler.load Worker.Default

                    use pdistShortMap = pdistShortMap |> Compiler.load Worker.Default
                    let run epochs = 
                        let nodes = this.InputNodes

                        let nNodes = nodes.Count
                        let nodeLen = nodes.[0].Count()

                        let mapLen = this.asArray.Length / nodeLen
                        let mapFullLen = this.asArray.Length

                        let len = if nNodes <= mapLen then this.asArray.Length else nNodes * nodeLen

                        let nt =  ((this.DimX * this.DimY) / nodeLen) * nodeLen
                        let nBlocks = this.GetBlockDim len nt //split the array of nodes into blocks

                        use dMap = m.Worker.Malloc(this.toArray)
                        let minDist = Array.create nNodes Double.MaxValue
                        use dMinDists = if nNodes <= mapLen then m.Worker.Malloc<float>(mapLen) else m.Worker.Malloc(minDist)
                        use dMinIndex = m.Worker.Malloc<int>(if nNodes <= mapLen then mapLen else nNodes) 
                        use dTemp = m.Worker.Malloc<float>(len)
                        use dNodes = m.Worker.Malloc(nodes.SelectMany(fun n -> n :> float seq).ToArray())

                        for epoch = 0 to epochs - 1 do
                            tic()

                            let mins = 
                                if nNodes <= mapLen  then
                                    pdist.Run dMap dNodes dTemp dMinDists dMinIndex mapFullLen nodeLen nNodes nt nBlocks
                                else
                                    pdistShortMap.Run dMap dNodes dTemp dMinDists dMinIndex mapFullLen nodeLen nNodes nt nBlocks

                            let nrule = this.ModifyTrainRule (float epoch) epochs
                            //printfn "Found all minimums, epoch %d, %10.3fms" epoch (toc())
                            //tic()
                            mins |> Seq.iteri 
                                (fun i bmu ->
                                    let (xBmu, yBmu) = this.toSomCoordinates bmu
                                    let mapNode = this.somMap.[xBmu, yBmu]
                                    if not (String.IsNullOrEmpty(this.InputNodes.[i].Class)) then
                                        let y = if mapNode.Class = this.InputNodes.[i].Class then 1. else -1.                  
                                        this.trainNode this.somMap.[xBmu, yBmu] this.InputNodes.[i] (nrule * y)
                                )
                            printfn "Classifier train iteration, epoch %d, %10.3fms" epoch (toc())
                    run
                    )
            }

        member this.pPairwiseDistance = 
            cuda {
                let! pairwiseKernel = 
                    <@  fun len nodeLen
                            (nodes : deviceptr<float>)
                            (dist : deviceptr<float>)
                            ->    

                            let x = blockIdx.x * blockDim.x + threadIdx.x
                            let y = blockIdx.y * blockDim.y + threadIdx.y

                            if y > x then
                                let i = x * nodeLen
                                let j = y * nodeLen
                                let mutable distance = 0.
                                for k = i to i + nodeLen - 1 do
                                    let l = j + k - i
                                    distance <- distance + (nodes.[k] - nodes.[l]) * (nodes.[k] - nodes.[l])
                                dist.[x * len + y - (x + 1) * (x + 2) / 2] <- sqrt distance
                    @> 

                    |> Compiler.DefineKernel
                return Entry(
                    fun
                        (m : Program) ->
                        let run () = 
                            let pdist = m.Apply pairwiseKernel
                            let nodes = if this.InputNodes.Count > 10000 then this.InputNodes.Take(10000).ToList() else this.InputNodes

                            let len = nodes.Count
                            let nodeLen = nodes.[0].Count()

                            let nodes = nodes.SelectMany(fun (n: Node) -> n.AsEnumerable())
                            use dNodes = m.Worker.Malloc(nodes.ToArray())
                            use dDist = m.Worker.Malloc<float>(len * (len - 1) / 2)
                            let nt = dim3(min this.DimX len, min this.DimY len)
                            let nBlocks = dim3(this.GetBlockDim len nt.x, this.GetBlockDim len nt.y)
                            let lp = LaunchParam(nBlocks, nt)

                            pdist.Launch lp len nodeLen dNodes.Ptr dDist.Ptr
                            dDist.Gather()
                        run
                )
            }

        member this.pDensityMatrix =
            cuda {
                let! kernel =
                    <@ 
                        fun len nodeLen nNodes radius
                            (map : deviceptr<float>)
                            (nodes : deviceptr<float>)
                            (denseMatrix : deviceptr<int>)
                            ->

                            let i = blockIdx.x * blockDim.x + threadIdx.x

                            if i < len then
                                for n = 0 to nNodes - 1 do
                                    let mutable dist = 0.
                                    for k = 0 to nodeLen - 1 do
                                        dist <- dist + (nodes.[n*nodeLen + k] - map.[i * nodeLen + k]) * (nodes.[n*nodeLen + k] - map.[i * nodeLen + k])
                                    dist <- sqrt dist
                                    if dist < radius then
                                        denseMatrix.[i] <- denseMatrix.[i] + 1
                                        
                    @> |> Compiler.DefineKernel
                return Entry(
                    fun (m : Program)
                        
                         ->

                    let pdense = m.Apply kernel
                    let run radius = 
                        let len = this.Height * this.Width

                        let nodes = this.InputNodes.SelectMany(fun (n: Node) -> n.AsEnumerable()).ToArray()
                        use dMap = m.Worker.Malloc(this.toArray)
                        use dNodes = m.Worker.Malloc(nodes)
                        use dDense = m.Worker.Malloc(Array.zeroCreate len)
                        let nt = min (this.DimX * this.DimY) len
                        let nBlocks = this.GetBlockDim len nt
                        let lp = LaunchParam(nBlocks, nt)

                        let maxNodesPerRun = 15000

                        let iter = (this.InputNodes.Count + maxNodesPerRun) / maxNodesPerRun
                        let mutable prevNNodes = 0
                        for i = 1 to iter do
                            let nNodes = if this.InputNodes.Count - i * maxNodesPerRun < 0 then this.InputNodes.Count % maxNodesPerRun else maxNodesPerRun
                            pdense.Launch lp len this.NodeLen nNodes radius dMap.Ptr (dNodes.Ptr + i * prevNNodes) dDense.Ptr
                            prevNNodes <- nNodes
                            dDense.Gather() |> ignore
                        dDense.Gather()
                    run
                )
            }