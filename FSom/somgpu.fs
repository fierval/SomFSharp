namespace FSom

open System
open System.Linq
open System.Collections.Generic
open Alea.CUDA
open System.Diagnostics
open System.Threading
open System.Threading.Tasks
open System.Collections.Concurrent
open Microsoft.FSharp.Collections

[<AutoOpen>]
module SomGpuModule1 =

    type SomGpu1(dims, nodes) as this =
        inherit Som(dims, nodes)

        let getBlockDim len nThreads = (len + nThreads - 1) / nThreads
        let stopWatch = Stopwatch()
        let width, height = snd this.Dimensions, fst this.Dimensions

        let tic () = 
            stopWatch.Restart()

        let toc () = 
                stopWatch.Stop()
                stopWatch.Elapsed.TotalMilliseconds
        
        let somArray =
            let x, y = width, height
            let z = this.somMap.[0,0].Count()
            let arr : float [] ref = ref (Array.zeroCreate (x * y * z))
            this.somMap |> Array2D.iteri (fun i j e -> e |> Seq.iteri (fun k el -> (!arr).[i * x * z + z * j + k] <- el))
            !arr        
                
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
                    let R0 = float((max width height) / 2)
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

        // training
        let pDist = 
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
                    
                    //let streams = Array.init 2 (fun _ -> Engine.workers.DefaultWorker.CreateStream())
                    //let lps = streams |> Array.map(fun stream -> LaunchParam(nBlocks, nt, 0, stream))

                    // Step 1. Find the BMUs
                    let lp = LaunchParam(nBlocks, nt) //|> Engine.setDiagnoser diagnose
                    
                    kernel.Launch lp nodeLen len dNodes dMap.Ptr dDist.Ptr dAggDists.Ptr dMinDists.Ptr dMinIndex.Ptr
                    //kernel1.Launch m lps.[0] nodeLen len dNodes dMap.Ptr dDist.Ptr dMinDists.Ptr
                    //kernel2.Launch m lps.[1] nodeLen len (dNodes + nodeLen) dMap.Ptr dDist.Ptr dMinDists.Ptr

                    let minIndicies = dMinIndex.ToHost()
                    let minDists = dMinDists.ToHost()

                    let minVal = ref Double.MaxValue
                    let split = len / nodeLen

                    minDists |>Array.iteri (fun j e -> if e < !minVal then minVal := e; minIndex := minIndicies.[j])
                    !minIndex
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
                    let nBlocks = getBlockDim len nt 

                    use dMap = m.Worker.Malloc(somArray)
                    use dDist = m.Worker.Malloc<float>(len)
                    use dNodes = m.Worker.Malloc(nodes.SelectMany(fun n -> n :> float seq).ToArray())
                    use dAggDists = m.Worker.Malloc<float>(len / nodeLen)
                    use dMinDists = m.Worker.Malloc<float>(nBlocks)
                    use dMinIndex = m.Worker.Malloc<int>(nBlocks)

                    let lp = LaunchParam(nBlocks, nt) //|> Engine.setDiagnoser diagnose
                    let mins = Array.zeroCreate nNodes

                    for i in [0..nNodes - 1] do
                        mins.[i] <- pdist dMap dDist (dNodes.Ptr + i * nodeLen) dAggDists dMinDists dMinIndex nodeLen nNodes nt nBlocks
                        
                    mins
                    )
            }

        let pTrainSom = 
            cuda {
                let! pdist = pDist
                let! ptrain = pTrain
                
                return PFunc(fun (m: Module) (nodes : float [] list) epochs ->

                    let pdist = pdist.Apply m
                    let ptrain = ptrain.Apply m
                    let nNodes = nodes.Length

                    let nodeLen = nodes.[0].Length
                    let len = somArray.Length

                    let nt =  ((dimX * dimY) / nodeLen) * nodeLen
                    let nBlocks = getBlockDim len nt 

                    use dMap = m.Worker.Malloc(somArray)
                    use dDist = m.Worker.Malloc<float>(len)
                    use dNodes = m.Worker.Malloc(nodes.SelectMany(fun n -> n :> float seq).ToArray())
                    use dAggDists = m.Worker.Malloc<float>(len / nodeLen)
                    use dMinDists = m.Worker.Malloc<float>(nBlocks)
                    use dMinIndex = m.Worker.Malloc<int>(nBlocks)

                    let lp = LaunchParam(nBlocks, nt) //|> Engine.setDiagnoser diagnose

                    let mins = Array.zeroCreate nNodes
                    for epoch = 0 to epochs - 1 do 
                        // Step 1. Find the BMUs
                        tic ()

                        [0..nNodes - 1] 
                        |> Seq.iter(fun i -> mins.[i] <- pdist dMap dDist (dNodes.Ptr + i * nodeLen) dAggDists dMinDists dMinIndex nodeLen nNodes nt nBlocks)
                        printfn "found all minimums for the epoch: %10.3f ms" (toc())

                        tic()
                        //Step 2. Training.
                        ptrain epoch epochs mins nodeLen len dNodes.Ptr dMap.Ptr
                    dMap.ToHost()
                )    
            }
        member this.toArray = somArray

        member this.fromArray (somArray : float []) =
            let nodeLen = this.somMap.[0, 0].Count()
            let arr = Array.zeroCreate nodeLen
            for i = 0 to somArray.Length / nodeLen - 1 do    
                let x, y = this.toSomCoordinates i
                for j = 0 to nodeLen - 1 do
                    arr.[j] <- somArray.[i * nodeLen + j]
                this.somMap.[x,y] <- Node(arr)
                
        member this.SingleDimBmu (node : Node) =
            let arrNode = node.ToArray()
            let nodeLen = arrNode.Length
            let len = somArray.Length
              
            let nt =  ((dimX * dimY) / nodeLen) * nodeLen
            let nBlocks = getBlockDim len nt 

            let distances = Array.zeroCreate (somArray.Length)
            let min = ref Double.MaxValue
            let index = ref 0
            let aggDist = Array.create (somArray.Length / nodeLen) Double.MaxValue
            let minDist = Array.zeroCreate nBlocks
            let minIndex = Array.zeroCreate nBlocks

            for blockX = 0 to nBlocks - 1 do
                for threadX =  nt - 1 downto 0 do
                    let mapI = nt * blockX + threadX
                    // actual position of the node component in the map
                    if mapI < len then                    
                            
                        let nodeIndex = threadX % nodeLen

                        distances.[mapI] <- (somArray.[mapI] - arrNode.[nodeIndex]) * (somArray.[mapI] - arrNode.[nodeIndex])
                        if threadX % nodeLen = 0 then

                            // find the minimum among threads
                            let mutable sum = 0.
                            for j = mapI to mapI + nodeLen - 1 do
                                sum <- sum + distances.[j]
                            aggDist.[mapI / nodeLen] <- sum

                        // linger a bit longer to find local minimum
                        if threadX = 0 then
                            let mutable i = mapI / nodeLen
                            let mutable j = 0

                            while i < len / nodeLen && j < nt / nodeLen do
                                if aggDist.[i] < minDist.[blockX] || i = mapI / nodeLen then
                                    minDist.[blockX] <- aggDist.[i]
                                    minIndex.[blockX] <- i
                                i <- i + 1
                                j <- j + 1
                                    

            let minVal = ref Double.MaxValue
            minDist |> Array.iteri (fun j e -> if e < !minVal then minVal := e; index := minIndex.[j])
            !index
        
        member this.SingleDimTrain (node : Node) =
            let blockDim len nThreads = (len + nThreads - 1) / nThreads

            let nodeLen = nodes.First().Count()
            let len = somArray.Length

            let R0 = float((max width height) / 2)
            let nrule0 = 0.9

            let dim = min width (int(sqrt(float (dimX * dimY / nodeLen))))
            let nt =  dim3(dim, dim, nodeLen)
            let nBlocks = dim3(blockDim width nt.x, blockDim height nt.y, 1)

            let nodes = nodes.SelectMany(fun n -> n :> float seq).ToArray()
            
            let bmu = this.SingleDimBmu node
            for blockX = 0 to nBlocks.x - 1 do
                for blockY = 0 to nBlocks.y - 1 do
                    for threadX = 0 to nt.x - 1 do
                        for threadY = 0 to nt.y - 1 do
                            for threadZ = 0 to nt.z - 1 do
                                let bmuX, bmuY = this.toSomCoordinates bmu
                                let rSq = R0 * R0
                                let x = nt.x * blockX + threadX 
                                let y = nt.y * blockY + threadY

                                let i = x * width * nodeLen + y * nodeLen + threadZ

                                if i < len then
                                    let distSq = float((bmuX - x) * (bmuX - x) + (bmuY - y) * (bmuY - y))
                                    if distSq < rSq then 
                                        somArray.[i] <- somArray.[i] + nrule0 * exp(-(1.0 * distSq) / (rSq)) * (node.[threadZ] - somArray.[i])


        member this.toSomCoordinates i =
            let x = i / width 
            let y = i - x * width
            x, y
        
        member this.Train epochs =
            let worker = Engine.workers.DefaultWorker
            use pfuncm = worker.LoadPModule(pTrainSom)

            let somArray = pfuncm.Invoke (nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList)  epochs
            this.fromArray somArray
        
        member this.GetBmuGpu (nodes : seq<Node>)  =
            let worker = Engine.workers.DefaultWorker
            use pfuncm = worker.LoadPModule(pTestBmu)

            let res = pfuncm.Invoke ((nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList))
            res

        member this.MergeNodes () =
            nodes.SelectMany(fun (n : Node) -> n :> IEnumerable<float>)