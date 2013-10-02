﻿open FSom
open System.Linq
open System.Collections.Generic
open System
open System.Diagnostics

let stopWatch = Stopwatch()

let tic () = 
    stopWatch.Restart()

let toc () = 
        stopWatch.Stop()
        stopWatch.Elapsed.TotalMilliseconds

let mainTainTest argv = 
    let bound = 12000
    let nodes = ([1..bound] |> Seq.map (fun i -> Node(12))).ToList()
    let som1 = SomGpu1((200, 200), nodes)
    let som2 = SomGpu2((200, 200), nodes)

//    printfn "training with node-by-node bmu\n"
//    som1.Train 5 3

    printfn "training with massive iterations bmu\n"
    som2.Train 5 |> ignore

    0

//[<EntryPoint>]
let main argv = 
    
    for j = 3 to 4 do
        let upper = 10. ** float(j)
        let bound = int(floor(upper * 1.2))
        let nodes = ([1..bound] |> Seq.map (fun i -> Node(12))).ToList()
        let som1 = SomGpu1((200, 200), nodes)
        let som2 = SomGpu2((200, 200), nodes)

        printfn "\n"
        printfn "Number of nodes: %d" bound
        for i = 1 to 3 do
            printfn "\n"
            printfn "============================"
            printfn "\tAttempt: %d" i

            tic()
            let minsGpu2 = som2.GetBmuGpu nodes
            printfn "\tgpu iterations multiple copies of nodes: %10.3f ms" (toc())

            tic()
            let minsGpu1 = som1.GetBmuGpu nodes 
            printfn "\tgpu node-by-node: %10.3f ms" (toc())


            if j < 2 then
                tic()
                nodes |> Seq.iter( fun node -> som1.GetBMUParallel(node) |> ignore)
                printfn "\tcpu parallel: %10.3f ms" (toc())

                tic()
                nodes |> Seq.iter( fun node -> som1.GetBMU(node) |> ignore)
                printfn "\tcpu sequential: %10.3f ms" (toc())

                let rnd = Random(int(DateTime.Now.Ticks))
                let ind = rnd.Next(0, bound)
                tic()
                let min = som2.GetBMU(nodes.[ind])
                printfn "cpu timing for a single node: %10.3f ms" (toc())

let mainBmuTest argv = 
    let bound = 120
    let dim = 3
    let nodes = ([1..bound] |> Seq.map (fun i -> Node(11))).ToList()
    let som1 = SomGpu1((20, dim), nodes)
    //let som2 = SomGpu2((1, dim), nodes)
    //let bmus = som2.GetBmuGpuSingle nodes
    //let bmus = som2.GetBmuGpu nodes
    let bmus = som1.GetBmuGpu nodes
    let mutable failed = 0
    for i = 0 to bound - 1 do
        let bmu = som1.GetBMU nodes.[i]
        let bmuSom = som1.toSomCoordinates bmus.[i]
        //let index = som1.SingleDimBmu nodes.[i]
        //let bmuSom = som1.toSomCoordinates index
        
        if bmu = bmuSom then 
            () //printfn "Success!"
        else
            printfn "ind: %d, bmu: %A, bmuSom: %A" i bmu bmuSom
            failed <- failed + 1
    printfn "failed: %d" failed

let classifyTest argv = 
    let bound = 52000
    let dim1 = 8
    let dim2 = 8
    let nodes = ([1..bound] |> Seq.map (fun i -> Node(12))).ToList()
    let som1 = SomGpu1((dim1, dim2), nodes)
    for i = 1 to 3 do
        printfn "\n"
        printfn "============================"
        printfn "\tAttempt: %d" i

        tic()
        let minsGpu1 = som1.GetBmuGpu nodes 
        printfn "\tgpu node-by-node: %10.3f ms" (toc())


[<EntryPoint>]
let tests argv =
    classifyTest argv
    //mainTainTest argv
    //for i = 0 to 10 do
    //mainBmuTest argv
    0