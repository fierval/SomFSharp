namespace FSom

open System
open System.Linq
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

                            for thread  in [0..nodeLen..blockDim.x  - 1] do
                                if mapI + thread < len then 
                                    let mutable sum = 0.
                                    for j = 0 to nodeLen - 1 do
                                        sum <- sum + distances.[mapI + thread + j]
                                    mins.[(mapI + thread) / nodeLen] <- sqrt(sum)                @> 
                |> defineKernelFunc

            return PFunc(fun (m:Module) (node : float []) (map : float []) ->
                let kernel = kernel.Apply m
                let nodeLen = node.Length
                use dNode = m.Worker.Malloc(node)
                use dMap = m.Worker.Malloc(map)
                use dDist = m.Worker.Malloc<float>(map.Length)
                use dMins = m.Worker.Malloc<float>(map.Length / nodeLen)
                let nt = (1024 / nodeLen) * nodeLen // number of threads divisible by nodeLen
                let nBlocks = (map.Length + nt - 1)/ nt //map.Length is a multiple of nodeLen by construction
                let lp = LaunchParam(nBlocks, nt)
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