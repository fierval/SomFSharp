namespace FSom

open System
open System.Linq
open System.Collections.Generic
open Alea.CUDA
open System.Diagnostics
open System.Threading
open System.Threading.Tasks

[<AutoOpen>]
module SomGpuMassiveFullRotationModule =


    type SomGpu2(dims, nodes) as this =
        inherit Som(dims, nodes)
        
        let somArray =
            let x, y = dims
            let z = this.somMap.[0,0].Count()
            let arr : float [] ref = ref (Array.zeroCreate (x * y * z))
            this.somMap |> Array2D.iteri (fun i j e -> e |> Seq.iteri (fun k el -> (!arr).[i * x * z + z * j + k] <- el))
            !arr            

        let getBlockDim len nThreads = (len + nThreads - 1) / nThreads
        let stopWatch = Stopwatch()

        let tic () = 
            stopWatch.Restart()

        let toc () = 
                stopWatch.Stop()
                stopWatch.Elapsed.TotalMilliseconds

        let pDist = 
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
                        let nodeCell = cell / nodeLen % totalNodes

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

                                    __syncthreads()

                                    // first threads in the block wrap it up for everyone
                                    if nodeShift = 0 then
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
        let dimX, dimY = 32, 32

        let pTrain = 
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

                return PFunc(fun (m:Module) (epoch : int) epochs (mins : int []) nodeLen len (dNodes : DevicePtr<float>) (dMap : DevicePtr<float>) ->   
                    let kernelTrain = kernelTrain.Apply m
                 
                    // training constants                    
                    let width, height = fst this.Dimensions, snd this.Dimensions
                    let R0 = float(fst this.Dimensions / 2)
                    let nrule0 = 0.9
                    let modifyR x =
                        R0 * exp(-10.0 * (x * x) / float(epochs * epochs))
        
                    let modifyTrainRule x =
                            nrule0 * exp(-10.0 * (x * x) / float(epochs * epochs))

                    let dim = min width (int(sqrt(float (dimX * dimY / nodeLen))))
                    let nt =  dim3(dim, dim, nodeLen)

                    let nBlocks = dim3(getBlockDim width nt.x, getBlockDim height nt.y, 1)
                    let lp = LaunchParam(nBlocks, nt)

                    let r = modifyR (float epoch)
                    let nrule = modifyTrainRule (float epoch)

                    for i = 0 to mins.Length - 1 do
                        let bmu = mins.[i]
                        let bmuX, bmuY = this.toSomCoordinates bmu
                        kernelTrain.Launch lp nodeLen len width height bmu bmuX bmuY (r * r) nrule (dNodes + i * nodeLen) dMap
                    printfn "epoch: %d, nrule: %10.5f, R: %10.3f, time: %10.3f ms" epoch nrule r (toc())
                )            
            }

        let pTestBmu = 
            cuda {
                let! pdist = pDist

                return PFunc(
                        fun (m : Module) (nodes : float [] list) ->
                    let pdist = pdist.Apply m

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


                    pdist dMap dNodes dTemp dMinDists dIndex nodeLen nNodes nt nBlocks
                       
                    )
            }

        let pTrainSom = 
            cuda {
                let! pdist = pDist
                let! ptrain = pTrain
                
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


                    for epoch = 0 to epochs - 1 do 
                        // Step 1. Find the BMUs
                        tic ()

                        let mutable i = 0
                        let mins = pdist dMap dNodes dTemp dMinDists dIndex nodeLen nNodes nt nBlocks
                        printfn "found all minimums for the epoch: %10.3f ms" (toc())

                        tic()
                        //Step 2. Training.
                        ptrain epoch epochs mins nodeLen len dNodes.Ptr dMap.Ptr
                    dMap.ToHost()
                )    
            }

        member this.toArray = somArray
                    
        member this.toSomCoordinates i =
            let x = i / fst dims 
            let y = i - x * fst dims
            x, y
        
        member this.GetBmuGpuSingle (nodes : Node seq) =
            let len = somArray.Length
            let nodes = nodes.ToArray()
            let nodeLen = nodes.[0].Count()
            let totalNodes = nodes.Length
            let fit = len / nodeLen / totalNodes // how many times the nodes array "fits" the map array
            let mapLen = len / nodeLen

            let remainderCutoff = (len / nodeLen) %  totalNodes // iteration which will get redundant values for the part of the nodes array that doesn't "fit"

            let nt = (256 / nodeLen) * nodeLen // number of threads divisible by nodeLen
            let nBlocks = (len + nt - 1) / nt //split the array of nodes into blocks

            let minDist = Array.create mapLen Double.MaxValue
            let nodes = nodes.SelectMany(fun n -> n :> float seq).ToArray()
            let distances = Array.zeroCreate len
            let minIndex = Array.zeroCreate mapLen
            let finalMinIndex = Array.zeroCreate totalNodes

            for iter = 0 to len / nodeLen / fit - 1 do
                for block = 0 to nBlocks - 1 do
                    let shift = iter * nodeLen
                    let start = block * nt
                    for thread = nt - 1 downto 0 do
                        let mutable cell = start + thread
                        
                        let foldedNodeCell = cell / nodeLen // node "mapped" to the length of the map
                        let nodeCell = cell / nodeLen % totalNodes // actual node
                        
                        if cell < len then
                            cell <- cell + shift
                            if cell >= len then
                                cell <- cell - len
                            
                            if foldedNodeCell < totalNodes * fit || iter < remainderCutoff then
                                distances.[cell] <- (somArray.[cell] - nodes.[nodeCell * nodeLen + cell % nodeLen]) * (somArray.[cell] - nodes.[nodeCell * nodeLen + cell % nodeLen])

                                // first threads in the block wrap it up for everyone
                                if cell % nodeLen = 0 then
                                    let mapCell = cell / nodeLen
                                    let mutable distSq = distances.[cell]
                                    for j = 1 to nodeLen - 1 do
                                        distSq <- distSq + distances.[cell + j]
                                    if minDist.[foldedNodeCell] > distSq then
                                        minDist.[foldedNodeCell] <- distSq
                                        minIndex.[foldedNodeCell] <- mapCell

            for i in [0..totalNodes - 1] do
                let stride = totalNodes
                let mutable min = minDist.[i]
                finalMinIndex.[i] <- minIndex.[i]
                let mutable j = i + stride

                while j < mapLen do
                    if min > minDist.[j] then
                        min <- minDist.[j]
                        finalMinIndex.[i] <- minIndex.[j]
                    j <- j + stride
            finalMinIndex        

        member this.GetBmuGpu (nodes : Node seq) =
            let worker = Engine.workers.DefaultWorker
            use pfuncm = worker.LoadPModule(pTestBmu)

            let mins = pfuncm.Invoke (nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList)
            mins

        member this.MergeNodes () =
            nodes.SelectMany(fun (n : Node) -> n :> IEnumerable<float>)

        member this.Train epochs =
            let worker = Engine.workers.DefaultWorker
            use pfuncm = worker.LoadPModule(pTrainSom)

            pfuncm.Invoke (nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList) epochs
