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

    type SomGpu(dims, nodes) =
        inherit SomGpuBase(dims, nodes) 
        
        member this.fromArray () =
            let nodeLen = this.somMap.[0, 0].Count()
            let arr = Array.zeroCreate nodeLen
            for i = 0 to this.asArray.Length / nodeLen - 1 do    
                let x, y = this.toSomCoordinates i
                for j = 0 to nodeLen - 1 do
                    arr.[j] <- this.asArray.[i * nodeLen + j]
                this.somMap.[x,y] <- Node(arr)
                
        member this.Train epochs =
            let worker = Engine.workers.DefaultWorker
            use pfuncm = worker.LoadPModule(this.pTrainSom)

            pfuncm.Invoke (nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList) epochs
        
        member this.MergeNodes () =
            nodes.SelectMany(fun (n : Node) -> n :> IEnumerable<float>)