﻿namespace FSom

open System
open System.Linq
open System.Collections.Generic
open Alea.CUDA
open System.Diagnostics
open System.Threading
open System.Threading.Tasks

[<AutoOpen>]
module SomGpuModule1 =

    type SomGpu1(dims, nodes) as this =
        inherit Som(dims, nodes)
        
        let somArray =
            let x, y = dims
            let z = this.somMap.[0,0].Count()
            let arr : float [] ref = ref (Array.zeroCreate (x * y * z))
            this.somMap |> Array2D.iteri (fun i j e -> e |> Seq.iteri (fun k el -> (!arr).[i * x * z + z * j + k] <- el))
            !arr            
        let maxThreadsPerBlock = 256

        // training
        let pTrain = 
            cuda {
                let! kernel =
                    <@ fun nodeLen len 
                        (node : DevicePtr<float>) 
                        (map :  DevicePtr<float>)
                        (distances : DevicePtr<float>)
                        (minDist : DevicePtr<float>)
                        ->

                        // index into the original map, assuming
                        // a node is a single entity
                        let mapI = blockIdx.x * blockDim.x

                        // actual position of the node component in the map
                        let i = mapI + threadIdx.x  

                        if i < len then                    
                            // index into the node.
                            // mapI is divisible by nodeLen, so not necessary
                            // to include in this calculation
                            let j = threadIdx.x % nodeLen

                            distances.[i] <- (map.[i] - node.[j]) * (map.[i] - node.[j])
                            if i % nodeLen = 0 then
                                __syncthreads()

                                // find the minimum among threads
                                let mutable sum = 0.
                                for j = i to i + nodeLen - 1 do
                                    sum <- sum + distances.[j]
                                minDist.[i] <- sum
                                    
                        @> |> defineKernelFuncWithName "bmu"

                let! kernelTrain = 
                    <@ fun nodeLen len width height bmu bmuX bmuY rSq nRule
                        (node : DevicePtr<float>) 
                        (map :  DevicePtr<float>) 
                        ->

                        let x = blockDim.x * blockIdx.x + threadIdx.x 
                        let y = blockDim.y * blockIdx.y + threadIdx.y

                        let i = x * width * nodeLen + y * nodeLen

                        if i < len then
                            let distSq = float((bmuX - x) * (bmuX - x) + (bmuY - y) * (bmuY - y))
                            if distSq < rSq then
                                let y = exp(-(10.0 * distSq) / (rSq)) 
                                for j = i to i + nodeLen - 1 do
                                    map.[j] <- map.[j] + nRule * y * (node.[j - i] - map.[j])                                    

                    @> |> defineKernelFuncWithName "training"


                let diagnose (stats:KernelExecutionStats) =
                   printfn "gpu timing: %10.3f ms %6.2f%% threads(%d) reg(%d) smem(%d)"
                       stats.TimeSpan
                       (stats.Occupancy * 100.0)
                       stats.LaunchParam.BlockDim.Size
                       stats.Kernel.NumRegs
                       stats.Kernel.StaticSharedMemBytes

                let blockDim len nThreads = (len + nThreads - 1) / nThreads

                return PFunc(fun (m:Module) (nodes : float [] list) epochs ->
                    let kernel = kernel.Apply m
                    let kernelTrain = kernelTrain.Apply m

                    let nodeLen = nodes.[0].Length
                    let len = somArray.Length
                    let nt = (256 / nodeLen) * nodeLen // number of threads divisible by nodeLen
                    let nBlocks = blockDim len nt //map.Length is a multiple of nodeLen by construction
                    use dMap = m.Worker.Malloc(somArray)
                    use dDist = m.Worker.Malloc<float>(len)
                    use dNodes = m.Worker.Malloc(nodes.SelectMany(fun n -> n :> float seq).ToArray())
                    use dMinDists = m.Worker.Malloc<float>(len / nodeLen)

                    let dist = Array.create (len / nodeLen) Double.MaxValue

                    // training constants                    
                    let width, height = fst this.Dimensions, snd this.Dimensions
                    let R0 = float(fst this.Dimensions / 2)
                    let nrule0 = 0.9
                    let modifyR x =
                        R0 * exp(-10.0 * (x * x) / float(epochs * epochs))
        
                    let modifyTrainRule x =
                            nrule0 * exp(-10.0 * (x * x) / float(epochs * epochs))

                    let mins = (Array.zeroCreate nodes.Length)
                    for epoch = 0 to epochs - 1 do 
                        // Step 1. Find the BMUs
                        let lp = LaunchParam(nBlocks, nt) //|> Engine.setDiagnoser diagnose
                        nodes |> List.iteri (fun i node ->
                            kernel.Launch lp nodeLen len (dNodes.Ptr + i * nodeLen) dMap.Ptr dDist.Ptr dMinDists.Ptr
                            let minDists = dMinDists.ToHost()
                            let minVal = ref Double.MaxValue
                            minDists |> Array.iteri (fun j e -> if e < !minVal then minVal := e; mins.[i] <- j)
                            )
  
                        //Step 2. Training.
                        let nt = dim3(16, 16)
                        let nBlocks = dim3(blockDim width nt.x, blockDim height nt.y)
                        let lp = LaunchParam(nBlocks, nt)

                        let rSq = modifyR (float epoch)
                        let nrule = modifyTrainRule (float epoch)
                    
                        for i = 0 to nodes.Length - 1 do
                            let bmu = mins.[i]
                            let bmuX, bmuY = this.toSomCoordinates bmu
                            kernelTrain.Launch lp nodeLen len width height bmu bmuX bmuY rSq nrule (dNodes.Ptr + i * nodeLen) dMap.Ptr

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
                
        member this.SingleDimBmu blockSize (node : Node) =
            let arrNode = node.ToArray()
            let nodeLen = arrNode.Length
   
            let nt = (blockSize / nodeLen) * nodeLen // number of threads divisible by nodeLen
            let nBlocks = (somArray.Length + nt - 1)/ nt //map.Length is a multiple of nodeLen by construction
            let distances = Array.zeroCreate (somArray.Length)
            let mins = Array.zeroCreate (somArray.Length / nodeLen)
            let min = ref Double.MaxValue
            let index = ref 0
            let len = somArray.Length
            let minDist = Array.create (somArray.Length / nodeLen) Double.MaxValue

            for block = 0 to nBlocks - 1 do
                let mapI = block * nt
                for thread = nt - 1 downto 0 do
                        // actual position of the node component in the map
                        let i = mapI + thread  

                        if i < len then                    
                            // index into the node
                            let j = threadIdx.x % nodeLen

                            distances.[i] <- (somArray.[i] - node.[j]) * (somArray.[i] - node.[j])
                            if i % nodeLen = 0 then
                                // find the minimum among threads
                                let mutable sum = 0.
                                for j = i to i + nodeLen - 1 do
                                    sum <- sum + distances.[j]
                                minDist.[i] <- sum

            mins |> Array.iteri (fun i e -> if e < !min then min := e; index := i)
            !index
        
        member this.SingleDimTrain (node : Node) =
            let blockDim len nThreads = (len + nThreads - 1) / nThreads

            let width, height = fst this.Dimensions, snd this.Dimensions

            let R0 = float(fst this.Dimensions / 2)
            let nrule0 = 0.9

            let nt = dim3(16, 16)
            let nBlocks = dim3(blockDim width nt.x, blockDim height nt.y)
            let nodeLen = nodes.First().Count()
            let len = somArray.Length

            let nodes = nodes.SelectMany(fun n -> n :> float seq).ToArray()
            
            let bmu = this.SingleDimBmu maxThreadsPerBlock node
            for blockX = 0 to nBlocks.x - 1 do
                for blockY = 0 to nBlocks.y - 1 do
                    for threadX = 0 to nt.x - 1 do
                        for threadY = 0 to nt.y - 1 do
                            let bmuX, bmuY = this.toSomCoordinates bmu
                            let rSq = R0 * R0
                            let nRule = nrule0
                            let x = nBlocks.x * blockX + threadX 
                            let y = nBlocks.y * blockY + threadY

                            let i = x * width * nodeLen + y * nodeLen

                            if i < len then
                                let distSq = float((bmuX - x) * (bmuX - x) + (bmuY - y) * (bmuY - y))
                                if distSq < rSq then 
                                    let y = exp(-(10.0 * distSq) / (rSq))
                                    for j = i to i + nodeLen - 1 do
                                        somArray.[j] <- somArray.[j] + nRule * y * (node.[j - i] - somArray.[j])                                    


        member this.toSomCoordinates i =
            let x = i / fst dims 
            let y = i - x * fst dims
            x, y
        
        member this.Train epochs =
            let worker = Engine.workers.DefaultWorker
            use pfuncm = worker.LoadPModule(pTrain)

            let somArray = pfuncm.Invoke (nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList)  epochs
            this.fromArray somArray

        member this.MergeNodes () =
            nodes.SelectMany(fun (n : Node) -> n :> IEnumerable<float>)