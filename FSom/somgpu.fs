namespace FSom

open System
open System.Linq
open System.Collections.Generic
open Alea.CUDA
open System.Diagnostics

[<AutoOpen>]
module SomGpuModule =
    let pDistances = 
        cuda {
            let! kernel =
                <@ fun nodeLen len
                    (node : DevicePtr<float>) 
                    (map :  DevicePtr<float>)
                    (distances : DevicePtr<float>)
                    (mins : DevicePtr<float>) ->

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
                            let mutable frst = true
                            while mapI + thread < len && thread < blockDim.x do
                                let mutable sum = 0.
                                for j = 0 to nodeLen - 1 do
                                    sum <- sum + distances.[mapI + thread + j]
                                if frst || mins.[blockIdx.x] > sum then
                                    mins.[blockIdx.x] <- sum
                                thread <- thread + 3
                    @> 
                |> defineKernelFunc

            let diagnose (stats:KernelExecutionStats) =
               printfn "gpu timing: %10.3f ms %6.2f%% threads(%d) reg(%d) smem(%d)"
                   stats.TimeSpan
                   (stats.Occupancy * 100.0)
                   stats.LaunchParam.BlockDim.Size
                   stats.Kernel.NumRegs
                   stats.Kernel.StaticSharedMemBytes

            return PFunc(fun (m:Module) (node : float []) (map : float []) ->
                let kernel = kernel.Apply m
                let nodeLen = node.Length
                let nt = (256 / nodeLen) * nodeLen // number of threads divisible by nodeLen
                let nBlocks = (map.Length + nt - 1)/ nt //map.Length is a multiple of nodeLen by construction
                use dNode = m.Worker.Malloc(node)
                use dMap = m.Worker.Malloc(map)
                use dDist = m.Worker.Malloc<float>(map.Length)
                use dMins = m.Worker.Malloc<float>(nBlocks)
                let lp = LaunchParam(nBlocks, nt) //|> Engine.setDiagnoser diagnose
                kernel.Launch lp nodeLen (map.Length) dNode.Ptr dMap.Ptr dDist.Ptr dMins.Ptr
                dMins.ToHost()
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

        member this.GetBmuGpu (node : Node) = 
            let worker = Engine.workers.DefaultWorker
            use pfuncm = worker.LoadPModule(pDistances)
            let mins = pfuncm.Invoke (node.ToArray()) somArray
            mins

        member this.MergeNodes () =
            nodes.SelectMany(fun (n : Node) -> n :> IEnumerable<float>)