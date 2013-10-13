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
    let stopWatch = Stopwatch()

    let tic () = 
        stopWatch.Restart()

    let toc () = 
            stopWatch.Stop()
            stopWatch.Elapsed.TotalMilliseconds

    new (dim : int * int, fileName : string) as this = 
        SomGpu(dim, Som.Read fileName)  
        then this.ShouldClassify <- this.InputNodes.First(fun n-> not (String.IsNullOrEmpty(n.Class))).Count() > 0



    member this.fromArray (somArray : float []) =
        let nodeLen = this.somMap.[0, 0].Count()
        let arr = Array.zeroCreate nodeLen
        Parallel.For(0, somArray.Length / nodeLen, fun i ->
            let x, y = this.toSomCoordinates i
            for j = 0 to nodeLen - 1 do
                arr.[j] <- somArray.[i * nodeLen + j]
            this.somMap.[x,y] <- Node(arr)) |> ignore
        this.somMap
       
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

    member this.GetBmuGpuShortMap (nodes : Node seq) =
        let worker = Engine.workers.DefaultWorker
        use pfuncm = worker.LoadPModule(this.pTestDistShortMap)

        let mins = pfuncm.Invoke (nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList)
        mins
                            
    override this.Train epochs =
        let worker = Engine.workers.DefaultWorker
        use pfuncm = worker.LoadPModule(this.pTrainSom)

        pfuncm.Invoke (this.InputNodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList) epochs
        |> this.fromArray

    override this.TrainClassifier epochs =
        let classes = this.InitClasses()

        for epoch = 0 to epochs - 1 do
            tic()
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
            printfn "Classifier train iteration, epoch %d, %10.3fms" epoch (toc())

    member this.MergeNodes () =
        nodes.SelectMany(fun (n : Node) -> n :> IEnumerable<float>)

    override this.DistanceMap () =
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


