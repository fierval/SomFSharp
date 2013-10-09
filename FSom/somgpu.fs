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
open System.IO

type SomGpu(dims, nodes : Node seq) =
    inherit SomGpuBase(dims, nodes) 
    new (dim : int * int, fileName : string) = SomGpu(dim, Som.Read fileName)

    member this.fromArray (somArray : float []) =
        let nodeLen = this.somMap.[0, 0].Count()
        let arr = Array.zeroCreate nodeLen
        Parallel.For(0, somArray.Length / nodeLen, fun i ->
            let x, y = this.toSomCoordinates i
            for j = 0 to nodeLen - 1 do
                arr.[j] <- somArray.[i * nodeLen + j]
            this.somMap.[x,y] <- Node(arr)) |> ignore
        this.somMap
       
    member this.GetBmuGpuNodeByNode (nodes : seq<Node>)  =
        let worker = Engine.workers.DefaultWorker
        use pfuncm = worker.LoadPModule(this.pTestBmuNodeByNode)

        let res = pfuncm.Invoke ((nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList))
        res
    
    member this.GetBmuGpu (nodes : Node seq) =
        let worker = Engine.workers.DefaultWorker
        use pfuncm = worker.LoadPModule(this.pTestBmu)

        let mins = pfuncm.Invoke (nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList)
        mins

    member this.GetBmuGpuUnified (nodes : Node seq) =
        let worker = Engine.workers.DefaultWorker
        use pfuncm = worker.LoadPModule(this.pTestUnifiedBmu)

        let mins = pfuncm.Invoke (nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList)
        mins
                            
    member this.Train epochs =
        let worker = Engine.workers.DefaultWorker
        use pfuncm = worker.LoadPModule(this.pTrainSom)

        pfuncm.Invoke (nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList) epochs
        |> this.fromArray

    member this.TrainClassifier epochs =
        let classes = this.InitClasses()

        for epoch = 0 to epochs - 1 do
            let mins = this.GetBmuGpuUnified this.InputNodes
            let nrule = this.ModifyTrainRule (float epoch) epochs

            mins |> Seq.iteri 
                (fun i bmu ->
                    let (xBmu, yBmu) = this.toSomCoordinates bmu
                    let mapNode = this.somMap.[xBmu, yBmu]
                    if not (String.IsNullOrEmpty(this.InputNodes.[i].Class)) then
                        let y = if mapNode.Class = this.InputNodes.[i].Class then 1. else -1.                  
                        this.trainNode this.somMap.[xBmu, yBmu] this.InputNodes.[i] (nrule * y)
                )

    member this.MergeNodes () =
        nodes.SelectMany(fun (n : Node) -> n :> IEnumerable<float>)

    member this.DistanceMap () =
        let worker = Engine.workers.DefaultWorker
        use pfuncm = worker.LoadPModule(this.pDistanceMap)

        let map = pfuncm.Invoke

        // convert the single-dimensional map to two dimensions
        let distMap = 
            Array2D.init 
                (this.somMap |> Array2D.length1) 
                (this.somMap |> Array2D.length2) 
                (fun i j -> 
                    map.[i * this.Width + j]
                    )
        distMap

    member this.Save fileName =
        if String.IsNullOrWhiteSpace fileName then failwith "File name must be specified"

        if File.Exists fileName then File.Delete fileName

        let output = List<string>()

        let buildStringSeq (arr : string [,]) =
            seq {
                for i = 0 to this.Height - 1 do 
                    yield (arr.[i..i, 0..] |> Seq.cast<string> |> Seq.fold (fun st e -> st + " " + e) String.Empty)
            }
            
        // separator between chunks of output    
        let separate () = output.Add(String.Empty)

        // build the 2D array of distance map
        let distMap = this.DistanceMap()
        let strDistMap = distMap |> Array2D.map(fun e -> e.ToString()) |> buildStringSeq
        output.AddRange strDistMap

        separate()

        let classes = this.somMap |> Array2D.map (fun node -> node.Class) |> buildStringSeq
        output.AddRange classes
            
        separate()

        // 2D weights
        let weights = 
            Array2D.init this.Height (this.Width * this.NodeLen )
                (fun i j ->
                    this.somMap.[i, j / this.NodeLen].[ j % this.NodeLen].ToString()
                ) |> buildStringSeq

        output.AddRange weights

        // write it all out
        File.WriteAllLines(fileName, output)
            


