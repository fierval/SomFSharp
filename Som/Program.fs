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

[<EntryPoint>]
let main argv = 
    
    
    for j = 1 to 4 do
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
            let minsGpu1 = som1.Train 1
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
    
//                if som2.toSomCoordinates (minsGpu2.[ind]) = min then
//                    printfn "success!"
//                else
//                    printfn "needs work"
//

    0