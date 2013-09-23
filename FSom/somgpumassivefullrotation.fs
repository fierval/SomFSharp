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
    let maxThreadsPerBlock = 256

    let pDistancesMassive2 = 
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
                                    if minDist.[foldedNodeCell] > distSq then
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

            return PFunc(fun (m:Module) (nodes : float [] list) (map : float []) ->
                let kernel = kernel.Apply m
                let nodeLen = nodes.[0].Length
                let totalNodes = nodes.Length
                let len = map.Length
                let mapLen = len / nodeLen
                let nt = (maxThreadsPerBlock / nodeLen) * nodeLen // number of threads divisible by nodeLen
                let fit = len / nodeLen / totalNodes
                let nBlocks = (len + nt - 1) / nt //split the array of nodes into blocks

                use dMap = m.Worker.Malloc(map)
                let dist = Array.create mapLen Double.MaxValue
                use dMinDist = m.Worker.Malloc(dist)
                use dIndex = m.Worker.Malloc<int>(mapLen) 
                use dTemp = m.Worker.Malloc<float>(len)
                use dNodes = m.Worker.Malloc(nodes.SelectMany(fun n -> n :> float seq).ToArray())
                let lp = LaunchParam(nBlocks, nt) //|> Engine.setDiagnoser diagnose
                    
                for iter = 0 to len / nodeLen / fit - 1 do
                    kernel.Launch lp len nodeLen totalNodes iter dNodes.Ptr dMap.Ptr dTemp.Ptr dMinDist.Ptr dIndex.Ptr

                let finalMinIndex = Array.zeroCreate totalNodes
                
                let minDist = dMinDist.ToHost()
                let minIndex = dIndex.ToHost()

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
            )
        }

    type SomGpu2(dims, nodes) as this =
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
            use pfuncm = worker.LoadPModule(pDistancesMassive2)

            let mins = pfuncm.Invoke (nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList)  somArray
            mins

        member this.MergeNodes () =
            nodes.SelectMany(fun (n : Node) -> n :> IEnumerable<float>)
