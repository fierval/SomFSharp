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
    let maxThreadsPerBlock = 256

    let pDistancesMassive = 
        cuda {
            let! kernel =
                <@ fun len nodeLen totalNodes iter
                    
                    (nodes : DevicePtr<float>) 
                    (map :  DevicePtr<float>)
                    (distances : DevicePtr<float>)
                    (minDist : DevicePtr<float>) 
                    (minIndex : DevicePtr<int>)
                    ->
                    
                    let shift = iter * nodeLen
                    let start = blockIdx.x * blockDim.x
                    let cell = start + threadIdx.x

                    if cell < totalNodes * nodeLen then
                        let mutable mapCell = cell + shift
                        if mapCell >= len then
                            mapCell <- mapCell - len

                        distances.[cell] <- (map.[mapCell] - nodes.[cell]) * (map.[mapCell] - nodes.[cell])
                        __syncthreads()
                        if cell % nodeLen = 0 then
                            let mutable distSq = distances.[cell]
                            for j = 1 to nodeLen - 1 do
                                distSq <- distSq + distances.[cell + j]
                            let i = cell / nodeLen
                            if minDist.[i] > distSq then
                                minDist.[i] <- distSq
                                minIndex.[i] <- if i + iter < len / nodeLen then i + iter else i + iter - len/nodeLen

                @> |> defineKernelFunc

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
                let totalNodes = nodes.Length
                let len = map.Length
                let nt = (maxThreadsPerBlock / nodeLen) * nodeLen // number of threads divisible by nodeLen
                let nBlocks = (totalNodes * nodeLen + nt - 1)/ nt //split the array of nodes into blocks
                use dMap = m.Worker.Malloc(map)
                let dist = Array.create totalNodes Double.MaxValue
                use dMinDist = m.Worker.Malloc(dist)
                use dIndex = m.Worker.Malloc<int>(totalNodes) 
                use dTemp = m.Worker.Malloc(Array.zeroCreate (totalNodes * nodeLen))
                use dNodes = m.Worker.Malloc(nodes.SelectMany(fun n -> n :> float seq).ToArray())
                let lp = LaunchParam(nBlocks, nt) //|> Engine.setDiagnoser diagnose
                    
                for iter = 0 to len/nodeLen - 1 do
                    kernel.Launch lp len nodeLen totalNodes iter dNodes.Ptr dMap.Ptr dTemp.Ptr dMinDist.Ptr dIndex.Ptr
                dIndex.ToHost()
            )
        }

    type SomGpu(dims, nodes) as this =
        inherit Som(dims, nodes)
        
        let somArray =
            let x, y = dims
            let z = this.somMap.[0,0].Count()
            let arr : float [] ref = ref (Array.zeroCreate (x * y * z))
            this.somMap |> Array2D.iteri (fun i j e -> e |> Seq.iteri (fun k el -> (!arr).[i * x * z + z * j + k] <- el))
            !arr            

        member this.toArray = somArray
                    
        member this.toSomCoordinates i =
            let x = i / fst dims 
            let y = i - x * fst dims
            x, y
        
        member this.GetBmuGpu (nodes : Node seq) =
            let worker = Engine.workers.DefaultWorker
            use pfuncm = worker.LoadPModule(pDistancesMassive)

            let mins = pfuncm.Invoke (nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList)  somArray
            mins

        member this.MergeNodes () =
            nodes.SelectMany(fun (n : Node) -> n :> IEnumerable<float>)