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

        let tic () = 
            stopWatch.Restart()

        let toc () = 
                stopWatch.Stop()
                stopWatch.Elapsed.TotalMilliseconds
        
        let somArray =
            let x, y = dims
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

        // training
        let pDist = 
            cuda {
                let! kernel =
                    <@ fun nodeLen len 
                        (node : DevicePtr<float>) 
                        (map :  DevicePtr<float>)
                        (distances : DevicePtr<float>)
                        (minDist : DevicePtr<float>)
                        nodesAtOnce
                        ->

                        // index into the original map, assuming
                        // a node is a single entity
                        let i = blockDim.x * blockIdx.x + threadIdx.x 
                        let mapI = blockDim.x * blockIdx.x / nodesAtOnce + threadIdx.x
                        // actual position of the node component in the map
                        if mapI < len then                    
                            
                            let nodeIndex = blockIdx.x * blockDim.x / len

                            distances.[i] <- (map.[mapI] - node.[nodeIndex]) * (map.[mapI] - node.[nodeIndex])
                            if threadIdx.x % nodeLen = 0 then
                                __syncthreads()

                                // find the minimum among threads
                                let mutable sum = 0.
                                for j = i to i + nodeLen - 1 do
                                    sum <- sum + distances.[j]
                                minDist.[i / nodeLen / nodesAtOnce] <- sum
                                    
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
                                (dMinDists : DeviceMemory<float>) 
                                nodeLen nNodes (nt : int) 
                                (nBlocks : int) 
                                nodesAtOnce
                                ->

                    let kernel = kernel.Apply m
                    let len = somArray.Length
                    let minIndex = Array.zeroCreate nodesAtOnce
                    
                    //let streams = Array.init 2 (fun _ -> Engine.workers.DefaultWorker.CreateStream())
                    //let lps = streams |> Array.map(fun stream -> LaunchParam(nBlocks, nt, 0, stream))

                    // Step 1. Find the BMUs
                    let lp = LaunchParam(nBlocks, nt) //|> Engine.setDiagnoser diagnose
                    
                    kernel.Launch lp nodeLen len dNodes dMap.Ptr dDist.Ptr dMinDists.Ptr nodesAtOnce
                    //kernel1.Launch m lps.[0] nodeLen len dNodes dMap.Ptr dDist.Ptr dMinDists.Ptr
                    //kernel2.Launch m lps.[1] nodeLen len (dNodes + nodeLen) dMap.Ptr dDist.Ptr dMinDists.Ptr

                    let minDists = dMinDists.ToHost()
                    let minVal = Array.create nodesAtOnce Double.MaxValue
                    let split = len / nodeLen
                    minDists |>Array.iteri (fun j e -> if e < minVal.[j / split] then minVal.[j / split] <- e; minIndex.[j / split] <- j)
                    minIndex
                    )
            }
        
        let pTestBmu = 
            cuda {
                let! pdist = pDist

                return PFunc(
                        fun (m : Module) (nodes : float [] list) atOnce->
                    let pdist = pdist.Apply m

                    let nNodes = nodes.Length

                    let nodeLen = nodes.[0].Length
                    let len = somArray.Length

                    let nt =  ((dimX * dimY) / nodeLen) * nodeLen
                    let nBlocks = atOnce * getBlockDim len nt 

                    use dMap = m.Worker.Malloc(somArray)
                    use dDist = m.Worker.Malloc<float>(len * atOnce)
                    use dNodes = m.Worker.Malloc(nodes.SelectMany(fun n -> n :> float seq).ToArray())
                    use dMinDists = m.Worker.Malloc<float>(len / nodeLen * atOnce)

                    let lp = LaunchParam(nBlocks, nt) //|> Engine.setDiagnoser diagnose
                    let mins = Array.zeroCreate nNodes

                    for i in [0..atOnce..nNodes - 1] do
                        let curMins = pdist dMap dDist (dNodes.Ptr + i * nodeLen) dMinDists nodeLen nNodes nt nBlocks atOnce
                        
                        for k = 0 to atOnce - 1 do
                            mins.[i + k] <- curMins.[k]
                    mins
                    )
            }

        let pTrainSom = 
            cuda {
                let! pdist = pDist
                let! ptrain = pTrain
                
                return PFunc(fun (m: Module) (nodes : float [] list) epochs launchAtOnce ->
                    let width, height = fst this.Dimensions, snd this.Dimensions

                    let pdist = pdist.Apply m
                    let ptrain = ptrain.Apply m
                    let nNodes = nodes.Length

                    let nodeLen = nodes.[0].Length
                    let len = somArray.Length

                    let nt =  ((dimX * dimY) / nodeLen) * nodeLen
                    let nBlocks = launchAtOnce * getBlockDim len nt 

                    use dMap = m.Worker.Malloc(somArray)
                    use dDist = m.Worker.Malloc<float>(len * launchAtOnce)
                    use dNodes = m.Worker.Malloc(nodes.SelectMany(fun n -> n :> float seq).ToArray())
                    use dMinDists = m.Worker.Malloc<float>(len / nodeLen * launchAtOnce)
                    let lp = LaunchParam(nBlocks, nt) //|> Engine.setDiagnoser diagnose

                    let mins = Array.zeroCreate nNodes
                    for epoch = 0 to epochs - 1 do 
                        // Step 1. Find the BMUs
                        tic ()

                        let mutable i = 0
                        while i < nNodes - 1 do
                            let curMins = pdist dMap dDist (dNodes.Ptr + i * nodeLen) dMinDists nodeLen nNodes nt nBlocks launchAtOnce
                            for k = 0 to launchAtOnce - 1 do
                                mins.[i + k] <- curMins.[k]
                            i <- i + launchAtOnce
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
            let width, height = fst this.Dimensions, snd this.Dimensions
              
            let dim = min width (int(sqrt(float (dimX * dimY / nodeLen))))
            let nt =  dim3(dim, dim, nodeLen)
            let nBlocks = dim3(getBlockDim width nt.x, getBlockDim height nt.y, 1) //map.Length is a multiple of nodeLen by construction

            let distances = Array.zeroCreate (somArray.Length)
            let min = ref Double.MaxValue
            let index = ref 0
            let minDist = Array.create (somArray.Length / nodeLen) Double.MaxValue

            for blockX = 0 to nBlocks.x - 1 do
                for blockY = 0 to nBlocks.y - 1 do
                    for threadX = 0 to nt.x - 1 do
                        for threadY = 0 to nt.y - 1 do
                            for threadZ = nt.z - 1 downto 0 do
                                let x = nt.x * blockX + threadX 
                                let y = nt.y * blockY + threadY

                               // actual position of the node component in the map
                                let i = x * width * nodeLen + y * nodeLen + threadZ

                                if i < len then                    
                                    // index into the node
                                    distances.[i] <- (somArray.[i] - node.[threadZ]) * (somArray.[i] - node.[threadZ])
                                    if threadZ = 0 then
                                        // find the minimum among threads
                                        let mutable sum = 0.
                                        for j = i to i + nodeLen - 1 do
                                            sum <- sum + distances.[j]
                                        minDist.[i / nodeLen] <- sum

            let minVal = ref Double.MaxValue
            minDist |> Array.iteri (fun j e -> if e < !minVal then minVal := e; index := j)
            !index
        
        member this.SingleDimTrain (node : Node) =
            let blockDim len nThreads = (len + nThreads - 1) / nThreads

            let width, height = fst this.Dimensions, snd this.Dimensions
            let nodeLen = nodes.First().Count()
            let len = somArray.Length

            let R0 = float(fst this.Dimensions / 2)
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
            let x = i / fst dims 
            let y = i - x * fst dims
            x, y
        
        member this.Train epochs launchAtOnce =
            let worker = Engine.workers.DefaultWorker
            use pfuncm = worker.LoadPModule(pTrainSom)

            let pad = nodes.Count() % launchAtOnce
            let nodes = nodes |> Seq.append ([1..pad] |> Seq.map (fun _ -> Node(nodes.First().Count())))
            let somArray = pfuncm.Invoke (nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList)  epochs launchAtOnce
            this.fromArray somArray
        
        member this.GetBmuGpu (nodes : seq<Node>) atOnce =
            let worker = Engine.workers.DefaultWorker
            use pfuncm = worker.LoadPModule(pTestBmu)

            let pad = nodes.Count() % atOnce
            let nodes = nodes |> Seq.append ([1..pad] |> Seq.map (fun _ -> Node(nodes.First().Count())))

            let res = pfuncm.Invoke ((nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList)) atOnce
            res

        member this.MergeNodes () =
            nodes.SelectMany(fun (n : Node) -> n :> IEnumerable<float>)