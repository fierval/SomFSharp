namespace FSom
open Alea.CUDA
open System
open System.Diagnostics
open System.Linq
open System.Collections.Generic

[<AutoOpen>]
module kernels =
    type SomGpuBase(dims, nodes) as this =
        inherit Som(dims, nodes)
        let stopWatch = Stopwatch()

        let tic () = 
            stopWatch.Restart()

        let toc () = 
                stopWatch.Stop()
                stopWatch.Elapsed.TotalMilliseconds

        let getBlockDim len nThreads = (len + nThreads - 1) / nThreads
        let width, height = snd this.Dimensions, fst this.Dimensions

        let dimX, dimY = 32, 32
        
        let somArray =
            let x, y = width, height
            let z = this.somMap.[0,0].Count()
            let arr : float [] ref = ref (Array.zeroCreate (x * y * z))
            this.somMap |> Array2D.iteri (fun i j e -> e |> Seq.iteri (fun k el -> (!arr).[i * x * z + z * j + k] <- el))
            !arr        

        member this.pTrain = 
            cuda {
                let! kernelTrain = 
                    <@ fun nodeLen len width height bmu bmuX bmuY rSq nRule
                        (node : DevicePtr<float>) 
                        (map :  DevicePtr<float>) 
                        ->

                        let x = blockDim.x * blockIdx.x + threadIdx.x 
                        let y = blockDim.y * blockIdx.y + threadIdx.y
                        
                        let i = x * width * nodeLen + y * nodeLen + threadIdx.z

                        if i < len then
                            let distSq = float((bmuX - x) * (bmuX - x) + (bmuY - y) * (bmuY - y))
                            if distSq < rSq then
                                map.[i] <- map.[i] + nRule * exp(-(10.0 * distSq) / (rSq)) * (node.[threadIdx.z] - map.[i])                                    

                    @> |> defineKernelFuncWithName "training"

                return PFunc(fun (m:Module) (epoch : int) epochs r nrule (nt : dim3) (nBlocks : dim3) (mins : int []) nodeLen len (dNodes : DevicePtr<float>) (dMap : DevicePtr<float>) ->   
                    let kernelTrain = kernelTrain.Apply m
                    let lp = LaunchParam(nBlocks, nt)
                    for i = 0 to mins.Length - 1 do
                        let bmu = mins.[i]
                        let bmuX, bmuY = this.toSomCoordinates bmu
                        kernelTrain.Launch lp nodeLen len width height bmu bmuX bmuY (r * r) nrule (dNodes + i * nodeLen) dMap
                    printfn "epoch: %d, nrule: %10.5f, R: %10.3f, time: %10.3f ms" epoch nrule r (toc())
                )            
            }

        // training
        member this.pDistNodeByNode = 
            cuda {
                let! kernel =
                    <@ fun nodeLen len 
                        (node : DevicePtr<float>) 
                        (map :  DevicePtr<float>)
                        (distances : DevicePtr<float>)
                        (aggDist : DevicePtr<float>)
                        (minDist : DevicePtr<float>)
                        (minIndex : DevicePtr<int>)
                        ->

                        // index into the original map, assuming
                        // a node is a single entity
                        let mapI = blockDim.x * blockIdx.x + threadIdx.x
                        // actual position of the node component in the map
                        if mapI < len then                    
                            
                            let nodeIndex = threadIdx.x % nodeLen

                            distances.[mapI] <- (map.[mapI] - node.[nodeIndex]) * (map.[mapI] - node.[nodeIndex])
                            if threadIdx.x % nodeLen = 0 then
                                __syncthreads()

                                // find the minimum among threads
                                let mutable sum = 0.
                                for j = mapI to mapI + nodeLen - 1 do
                                    sum <- sum + distances.[j]
                                aggDist.[mapI / nodeLen] <- sum

                            // linger a bit longer to find local minimum
                            if threadIdx.x = 0 then
                                __syncthreads()
                                let mutable i = mapI / nodeLen
                                let mutable j = 0

                                while i < len / nodeLen && j < blockDim.x / nodeLen do
                                    if aggDist.[i] < minDist.[blockIdx.x] || i = mapI / nodeLen then
                                        minDist.[blockIdx.x] <- aggDist.[i]
                                        minIndex.[blockIdx.x] <- i
                                    i <- i + 1
                                    j <- j + 1
                                    
                        @>  |> defineKernelFuncWithName "bmu"
                

                let diagnose (stats:KernelExecutionStats) =
                    printfn "gpu timing: %10.3f ms %6.2f%% threads(%d) reg(%d) smem(%d)"
                        stats.TimeSpan
                        (stats.Occupancy * 100.0)
                        stats.LaunchParam.BlockDim.Size
                        stats.Kernel.NumRegs
                        stats.Kernel.StaticSharedMemBytes

                return 
                    PFunc(
                            fun (m:Module) 
                                (dMap : DeviceMemory<float>) 
                                (dDist : DeviceMemory<float>) 
                                (dNodes : DevicePtr<float>) 
                                (dAggDists : DeviceMemory<float>) 
                                (dMinDists : DeviceMemory<float>) 
                                (dMinIndex : DeviceMemory<int>)
                                nodeLen nNodes (nt : int) 
                                (nBlocks : int) 
                                
                                ->

                    let kernel = kernel.Apply m
                    let len = somArray.Length
                    let minIndex = ref 0
                    
                    // Step 1. Find the BMUs
                    let lp = LaunchParam(nBlocks, nt) //|> Engine.setDiagnoser diagnose
                    
                    kernel.Launch lp nodeLen len dNodes dMap.Ptr dDist.Ptr dAggDists.Ptr dMinDists.Ptr dMinIndex.Ptr

                    let minIndicies = dMinIndex.ToHost()
                    let minDists = dMinDists.ToHost()

                    let minVal = ref Double.MaxValue
                    let split = len / nodeLen

                    minDists |>Array.iteri (fun j e -> if e < !minVal then minVal := e; minIndex := minIndicies.[j])
                    !minIndex
                    )
            }

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
                        nodeLen 
                        nNodes 
                        (nt : int) 
                        (nBlocks : int) 
                        ->
                    let kernel = kernel.Apply m
                    let len = somArray.Length
                    let mapLen = len / nodeLen
                    let fit = len / nodeLen / nNodes

                    let lp = LaunchParam(nBlocks, nt) //|> Engine.setDiagnoser diagnose
                    
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

        member this.pTrainSom = 
            cuda {
                let! pdist = this.pDist
                let! ptrain = this.pTrain
                
                return PFunc(fun (m: Module) (nodes : float [] list) epochs ->
                    let width, height = fst this.Dimensions, snd this.Dimensions

                    let pdist = pdist.Apply m
                    let ptrain = ptrain.Apply m
                    let nNodes = nodes.Length
                    let nodeLen = nodes.[0].Length
                    let len = somArray.Length

                    let nt =  ((dimX * dimY) / nodeLen) * nodeLen
                    let nodeLen = nodes.[0].Length
                    let mapLen = len / nodeLen
                    let nt = (dimX * dimY / nodeLen) * nodeLen // number of threads divisible by nodeLen
                    let fit = len / nodeLen / nNodes
                    let nBlocks = getBlockDim len nt //split the array of nodes into blocks

                    use dMap = m.Worker.Malloc(somArray)
                    use dMinDists = m.Worker.Malloc<float>(mapLen)
                    use dIndex = m.Worker.Malloc<int>(mapLen) 
                    use dTemp = m.Worker.Malloc<float>(len)
                    use dNodes = m.Worker.Malloc(nodes.SelectMany(fun n -> n :> float seq).ToArray())

                    // training constants                    
                    let width, height = fst this.Dimensions, snd this.Dimensions
                    let R0 = float(fst this.Dimensions / 2)
                    let nrule0 = 0.9
                    let modifyR x =
                        R0 * exp(-10.0 * (x * x) / float(epochs * epochs))
        
                    let modifyTrainRule x =
                            nrule0 * exp(-10.0 * (x * x) / float(epochs * epochs))

                    let dim = min width (int(sqrt(float (dimX * dimY / nodeLen))))
                    let ntTrain =  dim3(dim, dim, nodeLen)

                    let nBlocksTrain = dim3(getBlockDim width ntTrain.x, getBlockDim height ntTrain.y, 1)

                    for epoch = 0 to epochs - 1 do 
                        // Step 1. Find the BMUs
                        tic ()

                        let mutable i = 0
                        let mins = pdist dMap dNodes dTemp dMinDists dIndex nodeLen nNodes nt nBlocks
                        printfn "found all minimums for the epoch: %10.3f ms" (toc())

                        let r = modifyR (float epoch)
                        let nrule = modifyTrainRule (float epoch)

                        tic()
                        //Step 2. Training.
                        ptrain epoch epochs r nrule ntTrain nBlocksTrain mins nodeLen len  dNodes.Ptr dMap.Ptr
                    dMap.ToHost()
                )    
            }
        member this.toSomCoordinates i =
            let x = i / width 
            let y = i - x * width
            x, y

        member this.asArray = somArray

        member this.GetBlockDim len nThreads = getBlockDim len nThreads
        member this.Width = width
        member this.Height = height
        member this.DimX = dimX
        member this.DimY = dimY
