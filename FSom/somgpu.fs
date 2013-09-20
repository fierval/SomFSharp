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
            let kernel =
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
                                thread <- thread + nodeLen
                    @> 

            let! kernel1 = kernel |> defineKernelFuncWithName "first"
            let! kernel2 = kernel |> defineKernelFuncWithName "second"
            let! kernel3 = kernel |> defineKernelFuncWithName "third"
            let! kernel4 = kernel |> defineKernelFuncWithName "fourth"
            let! kernel5 = kernel |> defineKernelFuncWithName "fifth"
            let! kernel6 = kernel |> defineKernelFuncWithName "sixths"
            let! kernel7 = kernel |> defineKernelFuncWithName "seventh"
            let! kernel8 = kernel |> defineKernelFuncWithName "eights"
            let! kernel9 = kernel |> defineKernelFuncWithName "nineth"
            let! kernel10 = kernel |> defineKernelFuncWithName "tenth"
            let! kernel11 = kernel |> defineKernelFuncWithName "eleventh"
            let! kernel12 = kernel |> defineKernelFuncWithName "twelveth"


            let diagnose (stats:KernelExecutionStats) =
               printfn "gpu timing: %10.3f ms %6.2f%% threads(%d) reg(%d) smem(%d)"
                   stats.TimeSpan
                   (stats.Occupancy * 100.0)
                   stats.LaunchParam.BlockDim.Size
                   stats.Kernel.NumRegs
                   stats.Kernel.StaticSharedMemBytes

            return PFunc(fun (m:Module) (streams:Stream []) (nodes : float [] list) (map : float []) ->
                //let kernel = kernel.Apply m
                let nodeLen = nodes.[0].Length
                let kernels = [kernel1; kernel2; kernel3; kernel4; kernel5; kernel6; kernel7; kernel8; kernel9; kernel10; kernel11; kernel12]
                let chunk = map.Length
                let nt = (256 / nodeLen) * nodeLen // number of threads divisible by nodeLen
                let nBlocks = (chunk + nt - 1)/ nt //map.Length is a multiple of nodeLen by construction
                use dMap = m.Worker.Malloc(map)
                use dDist = m.Worker.Malloc<float>(map.Length)
                use dMins = m.Worker.Malloc<float>(nBlocks * kernels.Length)
                let lps = streams |> Array.map (fun stream -> LaunchParam(nBlocks, nt, 0, stream) |> Engine.setDiagnoser diagnose) 
                nodes |> List.iteri (fun i node ->
                    use dNode = m.Worker.Malloc(node)
                    kernels.[i].Launch m lps.[i] nodeLen chunk dNode.Ptr dMap.Ptr dDist.Ptr (dMins.Ptr + i * nBlocks))
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

        member this.GetBmuGpu (nodes : Node seq) =
            let worker = Engine.workers.DefaultWorker
            use pfuncm = worker.LoadPModule(pDistances)
            let streams = Array.init 12 (fun _ -> worker.CreateStream())

            let mins = pfuncm.Invoke streams (nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList)  somArray
            mins

        member this.MergeNodes () =
            nodes.SelectMany(fun (n : Node) -> n :> IEnumerable<float>)