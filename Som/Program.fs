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
    let nodes = ([1..2] |> Seq.map (fun i -> Node(12))).ToList()
    let som = SomGpu((200, 200), nodes)

    for i = 1 to 3 do
        tic()

        nodes |> Seq.iter( fun node -> som.GetBmuGpu(node) |> ignore)
        printfn "gpu timing: %10.3f ms" (toc())

        tic()
        nodes |> Seq.iter( fun node -> som.GetBMUParallel(node) |> ignore)
        printfn "cpu timing: %10.3f ms" (toc())
    0