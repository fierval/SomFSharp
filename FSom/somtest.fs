﻿namespace FSom
open Alea.CUDA
open System
open System.Collections.Generic
open System.Linq

type SomGpuTest(dim, nodes : Node seq) =
    inherit Som(dim, nodes)

    new (dim : int * int, fileName : string) = SomGpuTest(dim, Som.Read fileName)

    member this.GetBmuGpuSingleNodeByNode (node : Node) =
        let arrNode = node.ToArray()
        let nodeLen = arrNode.Length
        let len = this.asArray.Length
              
        let nt =  ((this.DimX * this.DimY) / nodeLen) * nodeLen
        let nBlocks = this.GetBlockDim len nt 

        let distances = Array.zeroCreate (this.asArray.Length)
        let min = ref Double.MaxValue
        let index = ref 0
        let aggDist = Array.create (this.asArray.Length / nodeLen) Double.MaxValue
        let minDist = Array.zeroCreate nBlocks
        let minIndex = Array.zeroCreate nBlocks

        for blockX = 0 to nBlocks - 1 do
            for threadX =  nt - 1 downto 0 do
                let mapI = nt * blockX + threadX
                // actual position of the node component in the map
                if mapI < len then                    
                            
                    let nodeIndex = threadX % nodeLen

                    distances.[mapI] <- (this.asArray.[mapI] - arrNode.[nodeIndex]) * (this.asArray.[mapI] - arrNode.[nodeIndex])
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
        let len = this.asArray.Length

        let R0 = float((max this.Width this.Height) / 2)
        let nrule0 = 0.9

        let dim = min this.Width (int(sqrt(float (this.DimX * this.DimY / nodeLen))))
        let nt =  dim3(dim, dim, nodeLen)
        let nBlocks = dim3(blockDim this.Width nt.x, blockDim this.Height nt.y, 1)

        let nodes = nodes.SelectMany(fun n -> n :> float seq).ToArray()
            
        let bmu = this.GetBmuGpuSingleNodeByNode node
        for blockX = 0 to nBlocks.x - 1 do
            for blockY = 0 to nBlocks.y - 1 do
                for threadX = 0 to nt.x - 1 do
                    for threadY = 0 to nt.y - 1 do
                        for threadZ = 0 to nt.z - 1 do
                            let bmuX, bmuY = this.toSomCoordinates bmu
                            let rSq = R0 * R0
                            let x = nt.x * blockX + threadX 
                            let y = nt.y * blockY + threadY

                            let i = x * this.Width * nodeLen + y * nodeLen + threadZ

                            if i < len then
                                let distSq = float((bmuX - x) * (bmuX - x) + (bmuY - y) * (bmuY - y))
                                if distSq < rSq then 
                                    this.asArray.[i] <- this.asArray.[i] + nrule0 * exp(-(1.0 * distSq) / (rSq)) * (node.[threadZ] - this.asArray.[i])


    member this.GetBmuGpuSingle (nodes : Node seq) =
        let len = this.asArray.Length
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
                            distances.[cell] <- (this.asArray.[cell] - nodes.[nodeCell * nodeLen + cell % nodeLen]) * (this.asArray.[cell] - nodes.[nodeCell * nodeLen + cell % nodeLen])

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
   

    member this.GetDistanceMapSingle () =
        let len = this.asArray.Length
        let map = this.asArray
        let nodeLen = this.NodeLen
        let nt =  dim3(min this.Height this.DimX, min this.Width this.DimY)
        let mapLen = len / nodeLen
        let nBlocks = dim3(this.GetBlockDim mapLen nt.x, this.GetBlockDim mapLen nt.y)
        let distMap = Array.zeroCreate (this.Height * this.Width)

        for blockX = 0 to nBlocks.x - 1 do
            for blockY = 0 to nBlocks.y - 1 do
                for threadX = 0 to nt.x - 1 do
                    for threadY = 0 to nt.y - 1 do
                        let x = nBlocks.x * blockY + threadX
                        let y = nBlocks.y * blockY + threadY
                        
                        let i = x * this.Width * nodeLen + y * nodeLen
                        let mutable dist = 0.
                        let mutable n = 0

                        if i < this.Width * this.Height * nodeLen then 
                            let len = this.Width * this.Height
                            for x1 = x - 1 to x + 1 do
                                for y1 = y - 1 to y + 1 do
                                    if x1 >= 0 && y1 >= 0 && x1 < len && y1 < len && (x1 <> x || y1 <> y) then
                                        let j = x1 * this.Width * nodeLen + y1 * nodeLen
                                        n <- n + 1
                                        let mutable thisDist = 0.
                                        for z = 0 to nodeLen - 1 do
                                            thisDist <- thisDist + (map.[i + z] - map.[j + z]) * (map.[i + z] - map.[j + z])
                                        dist <- dist + sqrt thisDist

                        distMap.[i] <- dist / float(n)

        let distMap = 
            Array2D.init 
                (this.somMap |> Array2D.length1) 
                (this.somMap |> Array2D.length2) 
                (fun i j -> 
                    map.[i * this.Height + j]
                    )
        distMap

            
