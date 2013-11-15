namespace FSom

open System
open System.Linq
open System.Collections.Generic
open Alea.CUDA
open Alea.CUDA.Utilities
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

    new (dim : int * int, fileName : string, ?header) = 
        SomGpu(dim, Som.Read fileName (defaultArg header 0))  

    member this.fromArray (somArray : float []) =
        let nodeLen = this.somMap.[0, 0].Count()
        Parallel.For(0, somArray.Length / nodeLen, fun i ->
            let x, y = this.toSomCoordinates i
            let arr = Array.init nodeLen (fun j -> somArray.[i * nodeLen + j])
            this.somMap.[x,y] <- Node(arr)) |> ignore
        this.somMap
       
    member this.GetBmuGpu (nodes : Node seq) =
        use pfuncm = this.pTestBmu |> Compiler.load Worker.Default

        let mins = pfuncm.Run this.asArray (nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList)
        mins

    member this.GetBmuGpuUnified (map : float []) (nodes : Node seq) =
        use pfuncm = this.pTestUnifiedBmu |> Compiler.load Worker.Default

        let mins = pfuncm.Run map (nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList)
        mins

    member this.GetBmuGpuShortMap (nodes : Node seq) =
        use pfuncm = this.pTestDistShortMap |> Compiler.load Worker.Default

        let mins = pfuncm.Run this.asArray (nodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList)
        mins
                            
    override this.Train epochs =
        use pfuncm = this.pTrainSom |> Compiler.load Worker.Default

        printfn "starting to train on %d epochs" epochs
        
        this.somMap <- pfuncm.Run (this.InputNodes |> Seq.map (fun n -> n.ToArray()) |> Seq.toList) epochs
        |> this.fromArray
        this.somMap

    member this.MergeNodes () =
        this.InputNodes.SelectMany(fun (n : Node) -> n :> float seq)

    // initialize classes by assigning a class of the
    // nearest node to each code vector
    override this.InitClasses () =
        // for each code vector pick a node closest to it.
        // this is the reverse of finding a BMU.
        let map = this.MergeNodes().ToArray()
        let nodes = this.somMap |> Seq.cast<Node>
        let bmNodes = this.GetBmuGpuUnified map nodes
        let nodeLen = this.NodeLen

        nodes |> Seq.iteri
            (fun i node -> 
                let bmNode = bmNodes.[i]
                let x, y = this.toSomCoordinates i
                this.somMap.[x, y].Class <- this.InputNodes.[bmNode].Class
            )

    override this.TrainClassifier epochs =
        this.InitClasses()
        use pfuncm = this.pTrainClassifier |> Compiler.load Worker.Default
    
        pfuncm.Run epochs


    override this.DistanceMap () =
        use pfuncm = this.pDistanceMap |> Compiler.load Worker.Default

        let map = pfuncm.Run ()

        // convert the single-dimensional map to two dimensions
        let distMap = 
            Array2D.init 
                (this.somMap |> Array2D.length1) 
                (this.somMap |> Array2D.length2) 
                (fun i j -> 
                    map.[i * this.Width + j]
                    )
        distMap
    
    override this.Classify nodes =
        let mins = this.GetBmuGpuUnified this.asArray nodes
        mins |> Array.map 
            (fun m -> 
                let x, y = this.toSomCoordinates m
                this.somMap.[x, y].Class    
            )

    override this.PairwiseDistance () =
        use pfuncm = this.pPairwiseDistance |> Compiler.load Worker.Default

        pfuncm.Run ()

    override this.DensityMatrix () =
        use pfuncm = this.pDensityMatrix |> Compiler.load Worker.Default
        let radius = this.ParetoRadius

        let density = pfuncm.Run radius
        let arr = 
            Array2D.init this.Height this.Width 
                (fun i j ->
                    density.[i * this.Height + j])
        arr
            
            