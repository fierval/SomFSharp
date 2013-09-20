namespace FSom

open System
open System.Linq
open System.Collections.Generic
open Alea.CUDA
open System.Diagnostics
open System.Threading
open System.Threading.Tasks

[<AutoOpen>]
module SomGpuModule =
    let pMinimum = 
        cuda {
            let! kernel = 
                <@ 
                    fun nBlocks 
                        (minDist : DevicePtr<float>)
                        (minIndex : DevicePtr<int>) 
                        (mins : DevicePtr<int>) ->

                        let nodeBaseI = blockIdx.x * blockDim.x
                        let mutable min = minDist.[nodeBaseI]
                        mins.[blockIdx.x] <- minIndex.[nodeBaseI]
                        for j = 1 to nBlocks - 1 do
                            if minDist.[nodeBaseI + j] < min then
                                min <-minDist.[nodeBaseI + j]
                                mins.[blockIdx.x] <- minIndex.[nodeBaseI + j]

                @> |> defineKernelFunc

            return PFunc(fun (m:Module) nBlocks (nNodes : int) minDist minIndex -> 
                let kernel = kernel.Apply m
                let dMins = m.Worker.Malloc<int>(nNodes)
                let lp = LaunchParam(nNodes, nBlocks)
                kernel.Launch lp nBlocks minDist minIndex dMins.Ptr
                dMins.ToHost()
            )
        }

    let pDistances = 
        cuda {
            let! kernel =
                <@ fun nodeLen len
                    (node : DevicePtr<float>) 
                    (map :  DevicePtr<float>)
                    (distances : DevicePtr<float>)
                    (minDist : DevicePtr<float>)
                    (minIndex : DevicePtr<int>) 
                    ->

                    // index into the original map, assuming
                    // a node is a single entity
                    let mapI = blockIdx.x * blockDim.x

                    // actual position of the node component in the map
                    let i = mapI + threadIdx.x  

                    if i < len then                    
                        // index into the node
                        let j = threadIdx.x % nodeLen

                        distances.[i] <- (map.[i] - node.[j]) * (map.[i] - node.[j])
                        if threadIdx.x = 0 then
                            __syncthreads()
                            let mutable thread = 0
                            minIndex.[blockIdx.x] <- -1
                            
                            // find the minimum among threads
                            while mapI + thread < len && thread < blockDim.x do
                                let k = mapI + thread
                                let mutable sum = 0.
                                for j = 0 to nodeLen - 1 do
                                    sum <- sum + distances.[k + j]
                                if minDist.[blockIdx.x] > sum || minIndex.[blockIdx.x] < 0 then
                                    minDist.[blockIdx.x] <- sum
                                    minIndex.[blockIdx.x] <- k / nodeLen
                                thread <- thread + nodeLen
                                    
                    @> |> defineKernelFunc
            let! pMinimumKernel = pMinimum

            let diagnose (stats:KernelExecutionStats) =
               printfn "gpu timing: %10.3f ms %6.2f%% threads(%d) reg(%d) smem(%d)"
                   stats.TimeSpan
                   (stats.Occupancy * 100.0)
                   stats.LaunchParam.BlockDim.Size
                   stats.Kernel.NumRegs
                   stats.Kernel.StaticSharedMemBytes

            return PFunc(fun (m:Module) (nodes : float [] list) (map : float []) ->
                let kernel = kernel.Apply m
                let nodeLen = nodes.[0].Length
                let chunk = map.Length
                let nt = (512 / nodeLen) * nodeLen // number of threads divisible by nodeLen
                let nBlocks = (chunk + nt - 1)/ nt //map.Length is a multiple of nodeLen by construction
                use dMap = m.Worker.Malloc(map)
                use dDist = m.Worker.Malloc<float>(map.Length)
                use dMinIndices = m.Worker.Malloc<int>(nBlocks * nodes.Length)
                use dMinDists = m.Worker.Malloc<float>(nBlocks * nodes.Length)
                use dNodes = m.Worker.Malloc(nodes.SelectMany(fun n -> n :> float seq).ToArray())
                
                let lp = LaunchParam(nBlocks, nt) //|> Engine.setDiagnoser diagnose
                nodes |> List.iteri (fun i node ->
                    kernel.Launch lp nodeLen chunk (dNodes.Ptr + i * nodeLen) dMap.Ptr dDist.Ptr (dMinDists.Ptr + i * nBlocks) (dMinIndices.Ptr + i * nBlocks))
              
                let pMin = pMinimumKernel.Apply m
                if nBlocks <= 1024 then
                    pMin nBlocks nodes.Length dMinDists.Ptr dMinIndices.Ptr
                else
                    let minDists = dMinDists.ToHost()                                        
                    let indices = dMinIndices.ToHost()
                    let mins = (Array.zeroCreate nodes.Length).ToList()

                    Parallel.For(0, nodes.Length, fun i ->
                        let baseI = i * nBlocks
                        let mutable min = minDists.[baseI]
                        mins.[i] <- indices.[baseI]
                        for j = 1 to nBlocks - 1 do
                            if minDists.[baseI + j] < min then
                                min <-minDists.[baseI + j]
                                mins.[i] <- indices.[baseI + j]
                    ) |> ignore
                    mins.ToArray()
                )
        }

    type SomGpu(dims : int * int, nodes : Node seq) as this =
        inherit Som(dims, nodes)
        
        let somArray =
            let x, y = dims
            let z = this.somMap.[0,0].Count()
            let arr : float [] ref = ref (Array.zeroCreate (x * y * z))
            this.somMap |> Array2D.iteri (fun i j e -> e |> Seq.iteri (fun k el -> (!arr).[i * x * z + z * j + k] <- el))
            !arr            

        member this.toArray = somArray
                
        member this.SingleDimBmu blockSize (node : Node) =
            let arrNode = node.ToArray()
            let nodeLen = arrNode.Length
            let nt = (blockSize / nodeLen) * nodeLen // number of threads divisible by nodeLen
            let nBlocks = (somArray.Length + nt - 1)/ nt //map.Length is a multiple of nodeLen by construction
            let distances = Array.zeroCreate somArray.Length
            let mins = Array.zeroCreate (somArray.Length / nodeLen)
            for block = 0 to nBlocks - 1 do
                let mapI = block * nt
                for thread = nt - 1 downto 0 do
                    
                    if mapI + thread < somArray.Length then
                        // actual position of the node component in the map
                        let i = mapI + thread  
                        let j = thread % arrNode.Length

                        distances.[i] <- (somArray.[i] - node.[j]) * (somArray.[i] - node.[j])

                for thread  in [0..nodeLen..nt  - 1] do
                    if mapI + thread < somArray.Length then 
                        let mutable sum = 0.
                        for j = 0 to nodeLen - 1 do
                            sum <- sum + distances.[mapI + thread + j]
                        mins.[(mapI + thread) / nodeLen] <- sqrt(sum)
            mins
                     
        member this.toSomCoordinates i =
            let x = i / fst dims 
            let y = i - x * fst dims
            x, y

        member this.GetBmuGpu (nodes : Node seq) =
            let worker = Engine.workers.DefaultWorker
            use pfuncm = worker.LoadPModule(pDistances)

            let mins = pfuncm.Invoke (nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList)  somArray
            mins

        member this.MergeNodes () =
            nodes.SelectMany(fun (n : Node) -> n :> IEnumerable<float>)