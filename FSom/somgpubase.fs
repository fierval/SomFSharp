namespace FSom
open Alea.CUDA
open System
open System.Diagnostics
open System.Linq
open System.Collections.Generic

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
                    (node : DevicePtr<float>) 
                    (map :  DevicePtr<float>) 
                    ->

                    let i = blockDim.x * blockIdx.x + threadIdx.x
                    if i < len then    
                        let x = i / width / nodeLen
                        let y = (i - x * width * nodeLen) / nodeLen
                        let z = i % nodeLen

                        let distSq = float((bmuX - x) * (bmuX - x) + (bmuY - y) * (bmuY - y))
                        if distSq < rSq then
                            map.[i] <- map.[i] + nRule * exp(-(1.0 * distSq) / (rSq)) * (node.[z] - map.[i])                                    

                @> |> defineKernelFuncWithName "training"

            return PFunc(fun (m:Module) (epoch : int) epochs r nrule (nt : int) nBlocks (mins : int []) nodeLen len (dNodes : DevicePtr<float>) (dMap : DevicePtr<float>) ->   
                let kernelTrain = kernelTrain.Apply m
                let lp = LaunchParam(nBlocks, nt)
                for i = 0 to mins.Length - 1 do
                    let bmu = mins.[i]
                    let bmuX, bmuY = this.toSomCoordinates bmu
                    if r > 1. then
                        kernelTrain.Launch lp nodeLen len this.Width this.Height bmuX bmuY (r * r) nrule (dNodes + i * nodeLen) dMap
                    else 
                        this.trainNode this.somMap.[bmuX, bmuY] this.InputNodes.[i] nrule
            )            
        }

    member this.pDistanceMap =
        cuda {
            let! kernelDistanceMap = 
                <@ fun nodeLen width height 
                    (map : DevicePtr<float>)
                    (distMap : DevicePtr<float>) ->
                        
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

                @> |> defineKernelFuncWithName "dist_map"
        return 

            PFunc (fun (m : Module) -> 
                let pDistMap = kernelDistanceMap.Apply m
                let len = this.asArray.Length
                let nodeLen = this.NodeLen
                let mapLen = len / nodeLen
                let nt =  min (this.DimX * this.DimY) mapLen
                let nBlocks = this.GetBlockDim mapLen nt

                use dMap = m.Worker.Malloc(this.toArray)
                use dDists = m.Worker.Malloc<float>(mapLen)
                    
                let lp = LaunchParam(nBlocks, nt) 

                pDistMap.Launch lp nodeLen this.Width this.Height dMap.Ptr dDists.Ptr
                dDists.ToHost()        
            )

        }
    // training
    member this.pDist = 
        cuda {
            let! kernel =
                <@ fun len nodeLen totalNodes iter
                    
                    (nodes : DevicePtr<float>) 
                    (map :  DevicePtr<float>)
                    (distances : DevicePtr<float>)
                    (minDist : DevicePtr<float>) 
                    (minIndex : DevicePtr<int>)
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
                @> |> defineKernelFunc

            let diagnose (stats:KernelExecutionStats) =
                printfn "gpu timing: %10.3f ms %6.2f%% threads(%d) reg(%d) smem(%d)"
                    stats.TimeSpan
                    (stats.Occupancy * 100.0)
                    stats.LaunchParam.BlockDim.Size
                    stats.Kernel.NumRegs
                    stats.Kernel.StaticSharedMemBytes

            return PFunc(
                fun 
                    (m:Module) 
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
                    ->
                let kernel = kernel.Apply m
                let mapLen = len / nodeLen

                let lp = LaunchParam(nBlocks, nt) //|> Engine.setDiagnoser diagnose

                let fit = len / nodeLen / nNodes                    
                for iter = 0 to len / nodeLen / fit - 1 do
                    kernel.Launch lp len nodeLen nNodes iter dNodes.Ptr dMap.Ptr dTemp.Ptr dMinDists.Ptr dIndex.Ptr

                let finalMinIndex = Array.zeroCreate nNodes
                
                let minDist = dMinDists.ToHost()
                let minIndex = dIndex.ToHost()

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
            )
        }
    
    member this.pDistShortMap =
        cuda {
            let! kernel = 
                <@  fun mapLen nodeLen totalNodes iter
                    
                        (nodes : DevicePtr<float>) 
                        (map :  DevicePtr<float>)
                        (distances : DevicePtr<float>)
                        (minDist : DevicePtr<float>) 
                        (minIndex : DevicePtr<int>) ->

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

                @> |> defineKernelFuncWithName "small_map_dist"

            return PFunc(
                fun 
                    (m:Module) 
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
                    ->
                let kernel = kernel.Apply m
                let mapLen = len / nodeLen

                let lp = LaunchParam(nBlocks, nt) //|> Engine.setDiagnoser diagnose
                    
                for iter = 0 to mapLen - 1 do
                    kernel.Launch lp mapLen nodeLen nNodes iter dNodes.Ptr dMap.Ptr dTemp.Ptr dMinDists.Ptr dIndex.Ptr
                
                dIndex.ToHost()
                )

        }
    member this.pTrainSom = 
        cuda {
            let! pdist = this.pDist
            let! pdistShortMap = this.pDistShortMap
            let! pTrain = this.pTrain

                
            return PFunc(fun (m: Module) (nodes : float [] list) epochs ->
                let pdist = pdist.Apply m
                let ptrain = pTrain.Apply m
                let pdistShortMap = pdistShortMap.Apply m

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
                let getMins () =
                    if nNodes <= mapLen  then
                        pdist dMap dNodes dTemp dMinDists dMinIndex mapFullLen nodeLen nNodes nt nBlocks
                    else
                        pdistShortMap dMap dNodes dTemp dMinDists dMinIndex mapFullLen nodeLen nNodes nt nBlocks
                for epoch = 0 to epochs - 1 do 
                    // Step 1. Find the BMUs
                    tic ()

                    let mins = getMins()
                    //printfn "found all minimums for the epoch: %10.3f ms" (toc())
                    //tic()

                    let r = modifyR (float epoch)
                    let nrule = modifyTrainRule (float epoch)

                    //Step 2. Training.
                    ptrain epoch epochs r nrule nt nBlocksTrain mins nodeLen mapFullLen dNodes.Ptr dMap.Ptr
                    printfn "epoch: %d, nrule: %10.5f, R: %10.3f, time: %10.3f ms" epoch nrule r (toc())


                dMap.ToHost()
            )    
        }

        member this.pTestBmu = 
            cuda {
                let! pdist = this.pDist

                return PFunc(
                        fun (m : Module) (map : float []) (nodes : float [] list) ->
                    let pdist = pdist.Apply m

                    let nNodes = nodes.Length
                    let nodeLen = nodes.[0].Length
                    let len = map.Length

                    let nt =  ((this.DimX * this.DimY) / nodeLen) * nodeLen
                    let nodeLen = nodes.[0].Length
                    let mapLen = len / nodeLen
                    let fit = len / nodeLen / nNodes
                    let nBlocks = this.GetBlockDim len nt //split the array of nodes into blocks

                    use dMap = m.Worker.Malloc(map)
                    use dMinDists = m.Worker.Malloc<float>(mapLen)
                    use dIndex = m.Worker.Malloc<int>(mapLen) 
                    use dTemp = m.Worker.Malloc<float>(len)
                    use dNodes = m.Worker.Malloc(nodes.SelectMany(fun n -> n :> float seq).ToArray())


                    pdist dMap dNodes dTemp dMinDists dIndex len nodeLen nNodes nt nBlocks
                       
                    )
            }

        member this.pTestUnifiedBmu = 
            cuda {
                let! pdist = this.pDist
                let! pdistShortMap = this.pDistShortMap

                return PFunc(
                        fun (m : Module) (map : float []) (nodes : float [] list) ->
                    let pdist = pdist.Apply m
                    let pdistShortMap = pdistShortMap.Apply m

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
                        pdist dMap dNodes dTemp dMinDists dMinIndex map.Length nodeLen nNodes nt nBlocks
                    else
                        pdistShortMap dMap dNodes dTemp dMinDists dMinIndex map.Length nodeLen nNodes nt nBlocks
                    )
            }

        member this.pTestDistShortMap = 
            cuda {
                let! pdist = this.pDistShortMap

                return PFunc(
                        fun (m : Module) (map : float []) (nodes : float [] list) ->
                    let pdist = pdist.Apply m

                    let nNodes = nodes.Length
                    let nodeLen = nodes.[0].Length
                    let len = nNodes * nodeLen

                    let nt =  ((this.DimX * this.DimY) / nodeLen) * nodeLen
                    let mapLen = map.Length / nodeLen
                    let nBlocks = this.GetBlockDim len nt //split the array of nodes into blocks

                    use dMap = m.Worker.Malloc(map)
                    let minDist = Array.create nNodes Double.MaxValue
                    use dMinDists = m.Worker.Malloc(minDist)
                    use dIndex = m.Worker.Malloc<int>(nNodes) 
                    use dTemp = m.Worker.Malloc<float>(len)
                    use dNodes = m.Worker.Malloc(nodes.SelectMany(fun n -> n :> float seq).ToArray())

                    pdist dMap dNodes dTemp dMinDists dIndex len nodeLen nNodes nt nBlocks
                    
                    )
            }
        
        member this.pTrainClassifier =
            cuda {
                let! pdist = this.pDist
                let! pdistShortMap = this.pDistShortMap

                return PFunc(
                        fun (m : Module) epochs ->
                    let pdist = pdist.Apply m
                    let pdistShortMap = pdistShortMap.Apply m
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
                                pdist dMap dNodes dTemp dMinDists dMinIndex mapFullLen nodeLen nNodes nt nBlocks
                            else
                                pdistShortMap dMap dNodes dTemp dMinDists dMinIndex mapFullLen nodeLen nNodes nt nBlocks

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

                    )
            }
