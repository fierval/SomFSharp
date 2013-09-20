open FSom
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
    let nodes = ([1..1000] |> Seq.map (fun i -> Node(12))).ToList()
    let som = SomGpu((200, 200), nodes)

    for i = 1 to 5 do
        tic()
        let minsGpu = som.GetBmuGpu nodes
        printfn "gpu timing: %10.3f ms" (toc())

        if nodes.Count < 1000 then
            tic()
            nodes |> Seq.iter( fun node -> som.GetBMUParallel(node) |> ignore)
            printfn "cpu timing: %10.3f ms" (toc())

            tic()
            let min = som.GetBMUParallel(nodes.[5])
            printfn "cpu timing for a single node: %10.3f ms" (toc())

            if som.toSomCoordinates (minsGpu.[5]) = min then
                printfn "success!"
            else
                printfn "needs work"


    0