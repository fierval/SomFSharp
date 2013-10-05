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

[<AutoOpen>]
module SomGpuModule =

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
                            
        member this.Train epochs shouldClassify =
            let worker = Engine.workers.DefaultWorker
            use pfuncm = worker.LoadPModule(this.pTrainSom)

            pfuncm.Invoke (nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList) epochs shouldClassify
            |> this.fromArray

        member this.MergeNodes () =
            nodes.SelectMany(fun (n : Node) -> n :> IEnumerable<float>)

        member this.DistanceMap () =
            let worker = Engine.workers.DefaultWorker
            use pfuncm = worker.LoadPModule(this.pDistanceMap)

            let map = pfuncm.Invoke
            let distMap = Array2D.zeroCreate (this.somMap |> Array2D.length1) (this.somMap |> Array2D.length2)
            map 
            |> Seq.iteri 
                (fun i e -> 
                    let x, y = this.toSomCoordinates i
                    distMap.[x, y] <- e
                )
            distMap

